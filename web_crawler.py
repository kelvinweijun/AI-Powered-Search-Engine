import asyncio
import aiohttp
from bs4 import BeautifulSoup
import logging
from typing import Dict, List, Optional, Set
from databases import Database
from sqlalchemy import create_engine, Column, Integer, String, Text, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.future import select
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nest_asyncio
from collections import deque
import re
import logging
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urlunparse
import dateutil.parser
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urlunparse
import json
import os
from datetime import datetime, timezone, timedelta
from PIL import Image
from io import BytesIO
from sqlalchemy.dialects.mysql import insert
import base64

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# Database configuration for MySQL
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "root",
    "database": "search_engine",
    "charset": "utf8mb4"
}

# Create SQLAlchemy Base
Base = declarative_base()

from sqlalchemy import Column, Integer, String, Text, DateTime, LargeBinary, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()
    
class CrawledPage(Base):
    __tablename__ = 'crawled_pages'

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255))
    url = Column(String(255), unique=True, index=True, nullable=False)
    content = Column(Text)
    snippet = Column(Text)
    timestamp = Column(Text)
    crawled_at = Column(DateTime, default=datetime.now)

    # One-to-many relationship: one page has many images
    images = relationship(
        "CrawledImage",
        back_populates="page",
        primaryjoin="CrawledPage.url == foreign(CrawledImage.page_url)",
        cascade="all, delete-orphan"
    )

class CrawledImage(Base):
    __tablename__ = 'crawled_images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    page_url = Column(String(255), ForeignKey("crawled_pages.url", ondelete="CASCADE"), nullable=False)
    url = Column(String(255), unique=True, index=True, nullable=False)
    alt_text = Column(Text)
    title = Column(Text)
    width = Column(String(50))
    height = Column(String(50))
    filename = Column(String(255))
    content_type = Column(String(100))
    data = Column(LargeBinary(length=(2**24)))
    file_size = Column(Integer)
    scraped_at = Column(DateTime, default=datetime.now)

    # Belongs-to relationship: image belongs to one page
    page = relationship(
        "CrawledPage",
        back_populates="images",
        primaryjoin="foreign(CrawledImage.page_url) == CrawledPage.url"
    )
# Database Manager Class
class DatabaseManager:
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.engine = None
        self.session_factory = None
    
    def connect(self) -> None:
        try:
            # Create connection URL
            url = f"mysql+pymysql://{self.config['user']}:{self.config['password']}@{self.config['host']}"
            
            # Create engine without database specified first to create database if needed
            temp_engine = create_engine(url)
            with temp_engine.connect() as conn:
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {self.config['database']}"))
            
            # Now create the engine with the database specified
            self.engine = create_engine(
                f"{url}/{self.config['database']}?charset={self.config['charset']}",
                pool_pre_ping=True,  # Verify connections before using them
                pool_recycle=3600,   # Recycle connections after 1 hour
                echo=False           # Set to True for SQL query logging
            )
            
            self.session_factory = sessionmaker(bind=self.engine)
            logging.info("Database connection established successfully")
        except SQLAlchemyError as e:
            logging.error(f"Error connecting to database: {e}")
            raise
    
    def close(self) -> None:
        if self.engine:
            self.engine.dispose()
            logging.info("Database connection closed")
    
    def create_tables_if_not_exist(self) -> None:
        try:
            # Create tables only if they don't exist
            Base.metadata.create_all(self.engine)
            logging.info("Tables created or verified successfully")
        except SQLAlchemyError as e:
            logging.error(f"Error creating tables: {e}")
            raise

# Async database setup with 'databases'
DATABASE_URL = f"mysql+asyncmy://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
database = Database(DATABASE_URL)

# Initialize SentenceTransformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index for storing embeddings
dimension = 384  # The dimension of the embeddings generated by the model
faiss_index = faiss.IndexFlatL2(dimension)

# Function to check if URL already exists in database
async def url_exists(url: str) -> bool:
    try:
        query = select(CrawledPage).where(CrawledPage.url == url)
        result = await database.fetch_one(query)
        return result is not None
    except Exception as e:
        logging.error(f"Error checking if URL exists: {e}")
        return False

# Async web crawler function
async def fetch_page(session, url, timeout_seconds=10):
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with session.get(url, timeout=timeout) as response:
            response.raise_for_status()  # Raise an exception for HTTP errors
            return await response.text()
    except asyncio.TimeoutError:
        logging.warning(f"Timeout while fetching {url}")
    except Exception as e:
        logging.error(f"Failed to fetch {url}: {e}")
    return None  # Ensure None is returned on failure

# Define a URL node structure to track parent-child relationships
class UrlNode:
    def __init__(self, url, depth, parent=None):
        self.url = url
        self.depth = depth
        self.parent = parent
        self.children = set()
        self.processed = False

    def add_child(self, child_url):
        self.children.add(child_url)
        
    def __str__(self):
        return f"URL: {self.url}, Depth: {self.depth}, Children: {len(self.children)}, Processed: {self.processed}"

async def crawl(start_url, max_depth=10):
    visited = set()
    url_nodes = {}  # Dictionary to track URL nodes by URL
    
    # Create root node
    root_node = UrlNode(start_url, 0)
    url_nodes[start_url] = root_node
    
    # Queue for BFS crawling with backtracking
    queue = deque([root_node])
    
    async with aiohttp.ClientSession() as session:
        while queue and len(visited) < 1000:  # Limit total URLs or use another stopping condition
            current_node = queue.popleft()
            current_url = current_node.url
            current_depth = current_node.depth
            
            if current_url in visited or current_depth >= max_depth:
                continue
                
            visited.add(current_url)
            logging.info(f"Processing URL: {current_url} (Depth: {current_depth})")
            
            # Process the current URL
            new_urls = await process_url(session, current_node, url_nodes, visited)
            
            # Mark the current node as processed
            current_node.processed = True
            
            # If no new URLs found and we've reached a dead end
            if not new_urls:
                logging.info(f"No more URLs found at {current_url}, backtracking...")
                
                # Backtrack to parent if available and find unprocessed siblings
                parent = current_node.parent
                while parent and all(url_nodes[child].processed for child in parent.children if child in url_nodes):
                    logging.info(f"All children of {parent.url} are processed, moving up...")
                    parent = parent.parent
                
                # If we found a parent with unprocessed children, add them to the queue
                if parent:
                    for child_url in parent.children:
                        if child_url in url_nodes and not url_nodes[child_url].processed and child_url not in [node.url for node in queue]:
                            queue.append(url_nodes[child_url])
                            logging.info(f"Backtracked and added sibling: {child_url}")
            
            # If we have new URLs, add them to the queue
            else:
                # Add new URLs to the queue
                for new_url in new_urls:
                    if new_url in url_nodes and not url_nodes[new_url].processed and new_url not in [node.url for node in queue]:
                        queue.append(url_nodes[new_url])
            
            # Add a small delay to avoid being blocked
            await asyncio.sleep(0.1)
            
        logging.info(f"Crawling completed. Visited {len(visited)} URLs.")

async def process_url(session, current_node, url_nodes, visited, max_depth=10):
    current_url = current_node.url
    current_depth = current_node.depth
    new_urls = set()
    
    try:
        # Check if the URL already exists in the database
        url_in_db = await url_exists(current_url)
        
        # Fetch the page content
        page_content = await fetch_page(session, current_url)
        if page_content is None:
            logging.warning(f"Could not fetch content for URL: {current_url}")
            return new_urls  # Return empty set if the page could not be fetched
        
        soup = BeautifulSoup(page_content, 'html.parser')
        
        # Only save content to database if it's not already there
        if not url_in_db:
            # Extract title with robust error handling
            title = extract_title(soup)
            
            # Extract clean content with better text processing
            content = extract_content(soup)
            
            # Extract a meaningful snippet using a prioritized approach
            snippet = extract_snippet(soup)
            
            # Extract timestamp with multiple fallback methods
            timestamp = extract_timestamp(soup, current_url)
            
            # Extract images from the page
            images = extract_images(soup, current_url)
            
            # Save to database and add to FAISS index only if meaningful content was extracted
            if content and len(content.strip()) > 20:  # Only store pages with actual content
                # Save page content to database
                page_id = await save_page_to_db(title, current_url, content, snippet, timestamp)
                
                # Save images to database if any were found
                if images:
                    await save_images_to_db(images, page_id, current_url)
                    logging.info(f"Saved {len(images)} images from: {current_url}")
                
                await add_to_faiss(title, current_url, content, snippet, timestamp)
                logging.info(f"Saved new content: {current_url} (Published: {timestamp})")
            else:
                logging.info(f"Skipping storage for URL with insufficient content: {current_url}")
        else:
            logging.info(f"Skipping content storage for already crawled URL: {current_url}")
        
        # Extract and add new links to the queue with better filtering
        new_urls = extract_and_filter_links(soup, current_url, current_node, current_depth, 
                                           url_nodes, visited, max_depth)
    
    except Exception as e:
        logging.error(f"Error processing URL {current_url}: {str(e)}", exc_info=True)
    
    return new_urls

def extract_images(soup, base_url):
    """
    Extract images from the page with their metadata.
    
    Args:
        soup: BeautifulSoup object of the page
        base_url: Base URL for resolving relative URLs
    
    Returns:
        list: List of dictionaries containing image data
    """
    images = []
    for img in soup.find_all('img'):
        try:
            # Get image URL
            img_url = img.get('src')
            if not img_url:
                continue
            
            # Skip tiny/tracking pixels
            if img.get('width') == '1' or img.get('height') == '1':
                continue
            
            # Resolve relative URLs
            if not (img_url.startswith('http://') or img_url.startswith('https://') or img_url.startswith('data:')):
                img_url = urljoin(base_url, img_url)
            
            # Get image alt text
            alt_text = img.get('alt', '')
            
            # Get image title if available
            title = img.get('title', '')
            
            # Get dimensions if available
            width = img.get('width', None)
            height = img.get('height', None)
            
            # Create image data dictionary
            image_data = {
                'url': img_url,
                'alt_text': alt_text,
                'title': title,
                'width': width,
                'height': height,
                'filename': os.path.basename(urlparse(img_url).path) if not img_url.startswith('data:') else 'data_uri_image'
            }
            
            images.append(image_data)
        except Exception as e:
            logging.warning(f"Error extracting image data: {str(e)}")
    
    return images

async def fetch_image_data(session, image_url, max_size_bytes=5*1024*1024):  # 5MB limit
    """
    Fetch image data from URL, resize it while preserving the aspect ratio,
    and return as bytes.
    
    Args:
        session: aiohttp ClientSession
        image_url: URL of the image
        max_size_bytes: Maximum allowed size for the image in bytes
    
    Returns:
        tuple: (image_data_bytes, content_type) or (None, None) if fetch fails
    """
    try:
        # Handle data URI schemes
        if image_url.startswith('data:'):
            try:
                # Split the data URI
                header, encoded = image_url.split(',', 1)
                
                # Extract content type
                content_type = header.split(':')[1].split(';')[0]
                
                # Decode base64 if needed
                if 'base64' in header:
                    image_data = base64.b64decode(encoded)
                else:
                    image_data = encoded.encode('utf-8')
                
                # Check size
                if len(image_data) > max_size_bytes:
                    logging.warning(f"Skipping large data URI image, size: {len(image_data)} bytes")
                    return None, None
                
                # Handle SVG specifically
                if content_type == 'image/svg+xml':
                    return image_data, content_type
                
                # For other image types, process as before
                with Image.open(BytesIO(image_data)) as img:
                    img = img.convert('RGB')
                    max_size = (800, 800)
                    img.thumbnail(max_size, Image.LANCZOS)
                    
                    with BytesIO() as output:
                        img.save(output, format="JPEG", quality=70, optimize=True)
                        resized_image_data = output.getvalue()
                    
                    return resized_image_data, 'image/jpeg'
            
            except Exception as e:
                logging.warning(f"Error processing data URI image: {e}")
                return None, None
        
        # Regular HTTP/HTTPS image fetching
        async with session.get(image_url, timeout=30) as response:
            if response.status == 200:
                # Check content length if available
                content_length = response.headers.get('Content-Length')
                if content_length and int(content_length) > max_size_bytes:
                    logging.warning(f"Skipping large image: {image_url}, size: {content_length} bytes")
                    return None, None
                
                # Read image data
                image_data = await response.read()
                content_type = response.headers.get('Content-Type', '')
                
                # Double check actual size
                if len(image_data) > max_size_bytes:
                    logging.warning(f"Skipping large image after download: {image_url}, size: {len(image_data)} bytes")
                    return None, None
                
                # Handle SVG
                if content_type == 'image/svg+xml':
                    return image_data, content_type
                
                # Process other image types
                with Image.open(BytesIO(image_data)) as img:
                    img = img.convert('RGB')
                    max_size = (800, 800)
                    img.thumbnail(max_size, Image.LANCZOS)
                    
                    with BytesIO() as output:
                        img.save(output, format="JPEG", quality=70, optimize=True)
                        resized_image_data = output.getvalue()
                    
                    return resized_image_data, 'image/jpeg'

            else:
                logging.warning(f"Failed to fetch image: {image_url}, status: {response.status}")
                return None, None
    
    except Exception as e:
        logging.warning(f"Error fetching image {image_url}: {str(e)}")
        return None, None

def extract_title(soup):
    """Extract the page title with fallback mechanisms."""
    try:
        # Try the standard title tag first
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
            if title:
                return title
        
        # Try common heading patterns if title tag is empty
        for heading in soup.find_all(['h1', 'h2'], limit=2):
            text = heading.get_text(strip=True)
            if text and len(text) < 200:  # Reasonable title length
                return text
        
        # Try meta tags
        meta_title = soup.find('meta', property='og:title') or soup.find('meta', attrs={'name': 'twitter:title'})
        if meta_title and meta_title.get('content'):
            return meta_title['content'].strip()
            
        return "No Title Available"
    except Exception as e:
        logging.warning(f"Error extracting title: {str(e)}")
        return "No Title Available"

def extract_content(soup):
    """Extract the main content with improved cleaning."""
    try:
        # Remove unwanted elements that typically contain non-content
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside', 
                                     'noscript', 'iframe', 'svg', 'form']):
            element.decompose()
        
        # Try to find the main content container
        main_container = None
        for container in soup.find_all(['article', 'main', 'div', 'section']):
            # Look for containers with lots of text and few links (typical for main content)
            text_length = len(container.get_text())
            links = container.find_all('a')
            link_text_length = sum(len(a.get_text()) for a in links)
            
            # Good content usually has more non-link text than link text
            if text_length > 500 and text_length > 3 * link_text_length:
                main_container = container
                break
        
        if main_container:
            content = main_container.get_text(separator=' ', strip=True)
        else:
            # Fallback to body content
            if soup.body:
                content = soup.body.get_text(separator=' ', strip=True)
            else:
                content = soup.get_text(separator=' ', strip=True)
        
        # Clean up the content
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        content = re.sub(r'[^\x00-\x7F]+', ' ', content)  # Remove non-ASCII chars
        
        return content.strip()
    except Exception as e:
        logging.warning(f"Error extracting content: {str(e)}")
        return ""

def extract_snippet(soup):
    """Extract a meaningful snippet using a prioritized approach."""
    try:
        snippet = None
        
        # 1. Try meta description first (usually the best summary)
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            desc = meta_desc['content'].strip()
            if len(desc) >= 30:
                return trim_snippet(desc)
        
        # 2. Try Open Graph description
        og_desc = soup.find('meta', property='og:description')
        if og_desc and og_desc.get('content'):
            desc = og_desc['content'].strip()
            if len(desc) >= 30:
                return trim_snippet(desc)
                
        # 3. Try to find article introductory paragraph
        for container in soup.find_all(['article', 'main', 'div.content', 'section.content', 'div.entry-content']):
            # Look for first substantial paragraph
            for p in container.find_all('p'):
                text = p.get_text(strip=True)
                # Avoid navigation text, copyright notices, etc.
                if len(text) >= 100 and not any(s in text.lower() for s in ['cookie', 'copyright', 'privacy', 'terms']):
                    return trim_snippet(text)
        
        # 4. Try any paragraph with substantial text
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if len(text) >= 100:
                return trim_snippet(text)
        
        # 5. Last resort - get first portion of page text
        body_text = soup.body.get_text(separator=' ', strip=True) if soup.body else soup.get_text(separator=' ', strip=True)
        if body_text:
            # Clean up the text
            body_text = re.sub(r'\s+', ' ', body_text) 
            return trim_snippet(body_text)
            
        return "No preview available"
    except Exception as e:
        logging.warning(f"Error extracting snippet: {str(e)}")
        return "No preview available"

def trim_snippet(text):
    """Trim and clean a snippet to an appropriate length."""
    # Clean the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate if too long, trying to end at a sentence
    if len(text) > 300:
        # Try to end at a sentence boundary
        shortened = text[:300]
        sentence_end = max(shortened.rfind('. '), shortened.rfind('! '), shortened.rfind('? '))
        
        if sentence_end > 150:  # Only truncate at sentence if we have a substantial snippet
            return text[:sentence_end+1].strip()
        return shortened.strip() + "..."
    
    return text.strip()

def extract_timestamp(soup, url):
    """Extract publication timestamp using multiple methods with fallbacks."""
    try:
        # 1. Check structured data (JSON-LD)
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                json_data = json.loads(script.string)
                # Handle both single objects and arrays of objects
                json_objects = [json_data] if isinstance(json_data, dict) else json_data if isinstance(json_data, list) else []
                
                for obj in json_objects:
                    # Check various schema.org timestamp fields
                    for field in ['datePublished', 'dateModified', 'dateCreated']:
                        if isinstance(obj, dict) and field in obj:
                            date_str = obj[field]
                            parsed_date = parse_date(date_str)
                            if parsed_date:
                                return parsed_date
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
        
        # 2. Check common meta tags
        meta_tags = [
            ('meta', {'property': 'article:published_time'}),
            ('meta', {'property': 'article:modified_time'}),
            ('meta', {'name': 'pubdate'}),
            ('meta', {'name': 'publishdate'}),
            ('meta', {'name': 'date'}),
            ('meta', {'name': 'DC.date.issued'}),
            ('meta', {'itemprop': 'datePublished'}),
            ('meta', {'itemprop': 'dateModified'}),
            ('meta', {'property': 'og:updated_time'})
        ]
        
        for tag, attrs in meta_tags:
            meta_elem = soup.find(tag, attrs)
            if meta_elem and meta_elem.get('content'):
                parsed_date = parse_date(meta_elem['content'])
                if parsed_date:
                    return parsed_date
        
        # 3. Look for time elements
        for time_tag in soup.find_all('time'):
            if time_tag.get('datetime'):
                parsed_date = parse_date(time_tag['datetime'])
                if parsed_date:
                    return parsed_date
            elif time_tag.string:
                parsed_date = parse_date(time_tag.string.strip())
                if parsed_date:
                    return parsed_date
        
        # 4. Look for common date patterns in HTML classes
        date_classes = ['date', 'time', 'published', 'updated', 'post-date', 'byline',
                        'dateline', 'timestamp', 'article-date', 'entry-date']
        
        for class_name in date_classes:
            elements = soup.find_all(class_=lambda c: c and class_name in c.lower())
            for element in elements:
                text = element.get_text().strip()
                parsed_date = parse_date(text)
                if parsed_date:
                    return parsed_date
        
        # 5. Try to extract from URL if it contains a date pattern
        url_date = extract_date_from_url(url)
        if url_date:
            return url_date
        
        # No timestamp found
        return None
    except Exception as e:
        logging.warning(f"Error extracting timestamp: {str(e)}")
        return None

def parse_date(date_string):
    """Parse date string into a standardized format."""
    if not date_string or not isinstance(date_string, str):
        return None
        
    # Clean the date string
    date_string = date_string.strip()
    
    # Try parsing with dateutil parser
    try:
        parsed_date = dateutil.parser.parse(date_string, fuzzy=True)
        # Format consistently as ISO
        return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, OverflowError, AttributeError):
        pass
    
    # Try common date formats with regex
    date_patterns = [
        # YYYY-MM-DD
        r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
        # DD-MM-YYYY or MM-DD-YYYY
        r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        # Month name, DD, YYYY
        r'([A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4})',
        # DD Month name YYYY
        r'(\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{4})'
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, date_string)
        if match:
            try:
                match_str = match.group(1)
                parsed_date = dateutil.parser.parse(match_str, fuzzy=True)
                return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, OverflowError):
                continue
    
    return None

def extract_date_from_url(url):
    """Extract date from URL if it follows common patterns."""
    # Common URL date patterns: /YYYY/MM/DD/ or /YYYY/MM/slug or /YYYY-MM-DD-slug
    url_path = urlparse(url).path
    
    # Pattern: /YYYY/MM/DD/
    year_month_day = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url_path)
    if year_month_day:
        year, month, day = year_month_day.groups()
        try:
            if 1800 < int(year) < 2100 and 0 < int(month) <= 12 and 0 < int(day) <= 31:
                return f"{year}-{month}-{day} 00:00:00"
        except ValueError:
            pass
    
    # Pattern: /YYYY-MM-DD-slug
    date_slug = re.search(r'/(\d{4}-\d{2}-\d{2})-', url_path)
    if date_slug:
        try:
            date_str = date_slug.group(1)
            parsed_date = dateutil.parser.parse(date_str)
            return parsed_date.strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, OverflowError):
            pass
    
    # Pattern: /YYYY/MM/slug
    year_month = re.search(r'/(\d{4})/(\d{2})/', url_path)
    if year_month:
        year, month = year_month.groups()
        try:
            if 1800 < int(year) < 2100 and 0 < int(month) <= 12:
                return f"{year}-{month}-01 00:00:00"
        except ValueError:
            pass
    
    return None

def extract_and_filter_links(soup, current_url, current_node, current_depth, url_nodes, visited, max_depth):
    """Extract and filter links with better URL handling."""
    new_urls = set()
    base_url = urlparse(current_url)
    
    if current_depth + 1 >= max_depth:
        return new_urls
        
    try:
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Skip empty links, anchors, javascript, mailto, etc.
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue
            
            # Resolve relative URLs
            if not href.startswith(('http://', 'https://')):
                href = urljoin(current_url, href)
            
            # Normalize URL
            href = normalize_url(href)
            
            # Skip URLs we've already seen or processed
            if href in visited:
                continue
                
            # Skip non-HTML content
            if is_likely_binary_content(href):
                continue
                
            # Skip URLs that are likely to be irrelevant
            if is_irrelevant_url(href, base_url):
                continue
            
            # Create new node and add to tracking structures
            if href not in url_nodes:
                new_node = UrlNode(href, current_depth + 1, current_node)
                url_nodes[href] = new_node
                current_node.add_child(href)
                new_urls.add(href)
                logging.debug(f"Added to queue at depth {current_depth + 1}: {href}")
    except Exception as e:
        logging.warning(f"Error extracting links from {current_url}: {str(e)}")
    
    return new_urls

def normalize_url(url):
    """Normalize a URL to avoid duplicates."""
    try:
        # Parse the URL
        parsed = urlparse(url)
        
        # Normalize the path (remove trailing slashes, etc.)
        path = parsed.path
        while path.endswith('/') and len(path) > 1:
            path = path[:-1]
            
        # Rebuild the URL without fragments and with normalized path
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            parsed.params,
            parsed.query,
            ''  # Remove fragment
        ))
        
        return normalized
    except Exception:
        return url  # Return original if normalization fails

def is_likely_binary_content(url):
    """Check if URL likely points to binary content."""
    binary_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.zip', '.rar', 
                         '.jpg', '.jpeg', '.png', '.gif', '.mp3', '.mp4', '.avi']
    
    for ext in binary_extensions:
        if url.lower().endswith(ext):
            return True
    return False

def is_irrelevant_url(url, base_url):
    """Check if URL is likely irrelevant for content crawling."""
    # Skip URLs with excessive query parameters (likely tracking or search results)
    if '?' in url and len(url.split('?')[1]) > 100:
        return True
        
    # Skip URLs from different domains if configured to stay on same domain
    # (Uncomment and adjust as needed for your specific requirements)
    # if base_url.netloc and urlparse(url).netloc != base_url.netloc:
    #     return True
        
    # Skip common irrelevant paths
    irrelevant_patterns = [
        '/login', '/signin', '/signup', '/register', '/cart', '/checkout',
        '/account', '/privacy', '/terms', '/contact', '/about', '/search',
        '/feed', '/rss', '/print/', '/share', '/comment'
    ]
    
    path = urlparse(url).path.lower()
    for pattern in irrelevant_patterns:
        if pattern in path:
            return True
            
    return False

# Async function to save the crawled page to the database
async def save_page_to_db(title, url, content, snippet, timestamp):
    try:
        # Save to DB
        query = CrawledPage.__table__.insert().values(
            title=title,
            url=url,
            content=content,
            snippet=snippet,
            timestamp=timestamp
        )
        async with database.transaction():
            await database.execute(query)
        logging.info(f"Successfully saved: {url}")
    except Exception as e:
        logging.error(f"Failed to save {url} to the database: {e}")

# Main async function to save image metadata and binary data
async def save_images_to_db(images, page_id, page_url):
    async with aiohttp.ClientSession() as session:
        for image in images:
            try:
                # Check if the image already exists
                image_exists = await image_exists_in_db(image['url'])

                # Fetch image binary data
                image_data, content_type = await fetch_image_data(session, image['url'])

                if image_data:
                    # Prepare query for upsert
                    stmt = insert(CrawledImage).values(
                        url=image['url'],
                        page_url=page_url,
                        alt_text=image.get('alt_text'),
                        title=image.get('title'),
                        width=image.get('width'),
                        height=image.get('height'),
                        filename=image.get('filename'),
                        content_type=content_type,
                        data=image_data,
                        file_size=len(image_data),
                        scraped_at=datetime.now(timezone.utc)
                    )

                    # Upsert - Update existing entry if URL already exists
                    upsert_stmt = stmt.on_duplicate_key_update(
                        alt_text=stmt.inserted.alt_text,
                        title=stmt.inserted.title,
                        width=stmt.inserted.width,
                        height=stmt.inserted.height,
                        filename=stmt.inserted.filename,
                        content_type=stmt.inserted.content_type,
                        data=stmt.inserted.data,
                        file_size=stmt.inserted.file_size,
                        scraped_at=stmt.inserted.scraped_at,
                        page_url=stmt.inserted.page_url,
                    )

                    # Insert in transaction
                    async with database.transaction():
                        await database.execute(upsert_stmt)

                    logging.info(f"Saved image: {image['url']}")
            except Exception as e:
                logging.error(f"Error saving image {image.get('url')} for page {page_id}: {e}")
                
async def save_image_to_db(image_metadata):
    """
    Save image metadata and binary data to the database.
    
    Args:
        image_metadata: Dictionary containing image metadata and binary data
        session: SQLAlchemy async session for transaction handling
    """
    try:
        # Insert the image record into the CrawledImage table
        new_image = CrawledImage(
            url=image_metadata['url'],
            alt_text=image_metadata['alt_text'],
            title=image_metadata['title'],
            width=image_metadata['width'],
            height=image_metadata['height'],
            filename=image_metadata['filename'],
            content_type=image_metadata['content_type'],
            data=image_metadata['data'],
            file_size=image_metadata['file_size'],
            scraped_at=image_metadata['scraped_at']
        )

        # Get the generated image_id
        image_id = new_image.id

        logging.info(f"Successfully saved image {image_metadata['url']} for page {image_metadata['page_id']}")
        
    except Exception as e:
        logging.error(f"Error saving image {image_metadata['url']} for page {image_metadata['page_id']}: {str(e)}")
        
async def image_exists_in_db(image_url):
    """
    Check if image already exists in database.
    
    Args:
        image_url: URL of the image
    
    Returns:
        bool: True if image exists, False otherwise
    """
    query = select(CrawledImage.id).where(CrawledImage.url == image_url)
    result = await database.fetch_one(query)
    return result is not None
            
# Function to add content to the FAISS index
async def add_to_faiss(title, url, content, snippet, timestamp):
    content_with_snippet = content + snippet
    embedding = embedding_model.encode([content_with_snippet])[0]  # Get the embedding for the content
    faiss_index.add(np.array([embedding], dtype=np.float32))  # Add the embedding to the FAISS index
    logging.info(f"Added to FAISS index: {title} - {url}")

# Function to load existing database entries into FAISS index
async def load_existing_entries_to_faiss():
    try:
        query = select(CrawledPage)
        results = await database.fetch_all(query)
        
        if results:
            logging.info(f"Loading {len(results)} existing entries into FAISS index")
            
            for result in results:
                embedding = embedding_model.encode([result['content']])[0]
                faiss_index.add(np.array([embedding], dtype=np.float32))
                
            logging.info(f"Successfully loaded {len(results)} entries into FAISS index")
    except Exception as e:
        logging.error(f"Error loading existing entries into FAISS: {e}")

# Search function using FAISS index
async def search(query, top_k=5):
    query_embedding = embedding_model.encode([query])[0]  # Get the embedding for the query
    _, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), top_k)
    
    results = []
    for idx in indices[0]:
        # Fetch the page details using its index
        query = select(CrawledPage).where(CrawledPage.id == idx)
        result = await database.fetch_one(query)
        if result:
            results.append({
                'title': result['title'],
                'url': result['url'],
                'snippet': result['snippet']
            })

    return results

# Run the crawl
async def main():
    # Create and initialize the database manager
    db_manager = DatabaseManager(DB_CONFIG)
    
    try:
        # Set up database and tables
        db_manager.connect()
        db_manager.create_tables_if_not_exist()
        
        # Connect to async database for operations
        await database.connect()
        logging.info("Connected to async database successfully")
        
        # Load existing database entries into FAISS index
        #await load_existing_entries_to_faiss()
        
        # Start crawling
        await crawl('https://www.asiaone.com/')  # Start crawling from this URL
        
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
    finally:
        # Clean up resources
        try:
            await database.disconnect()
            logging.info("Disconnected from async database")
        except Exception:
            pass
        
        try:
            db_manager.close()
        except Exception:
            pass

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())
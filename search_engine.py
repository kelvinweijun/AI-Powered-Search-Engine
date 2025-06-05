import asyncio
import logging
from http.server import SimpleHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
from jinja2 import Environment, FileSystemLoader
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from databases import Database
from web_crawler import CrawledPage, CrawledImage, DATABASE_URL
import nest_asyncio
from datetime import datetime, timedelta
import base64
import io
from typing import List, Dict, Any, Optional, Tuple
import cgi
import uuid
import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import json
import re

# Initialize variables for the search
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.IndexFlatL2(384)
database = Database(DATABASE_URL)
page_id_list = []  # Keep track of page ids in FAISS order

# Initialize image embedding model
image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
image_model.eval()
# Remove the classification layer
image_embedding_model = torch.nn.Sequential(*list(image_model.children())[:-1])
# Define image transformation pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create a separate FAISS index for images
image_faiss_index = faiss.IndexFlatL2(2048)  # ResNet50 outputs 2048-dimensional embeddings
image_id_list = []  # Keep track of image ids in FAISS order

# Image metadata dictionary to store classification info
image_metadata = {}  # Will store image_id -> metadata (including category)

# Temporary storage for uploaded images
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Jinja2 template setup
env = Environment(loader=FileSystemLoader("templates"))
index_template = env.get_template("index.html")  # The homepage template
results_template = env.get_template("results.html")  # The results page template
image_results_template = env.get_template("image_results.html")  # Template for image results
image_search_template = env.get_template("image_search.html")  # New template for image search upload form
image_search_results_template = env.get_template("image_search_results.html")  # New template for reverse image search results

# Define custom filter for formatting dates
def format_date_filter(value):
    """Format a timestamp for display."""
    if not value:
        return ""
    try:
        # Parse the timestamp
        dt = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        return dt.strftime('%b %d, %Y')  # Format as 'Apr 29, 2025'
    except Exception as e:
        logging.warning(f"Error formatting date '{value}': {str(e)}")
        return value  # Return as-is if formatting fails

# Add the custom filter to Jinja2 environment
env.filters['format_date'] = format_date_filter

# Function to detect image category
def classify_image_type(img_data, file_extension, mime_type):
    """
    Classify image into categories:
    - Photo
    - Illustration
    - Animated
    """
    # Check for animated images (GIF)
    if mime_type == "image/gif" or file_extension.lower() == ".gif":
        # Try to open as GIF and check number of frames
        try:
            img = Image.open(io.BytesIO(img_data))
            try:
                img.seek(1)  # Try to seek to second frame
                return "Animated"  # If it has multiple frames, it's animated
            except EOFError:
                # Only one frame, not animated
                pass
        except Exception as e:
            logging.warning(f"Error checking if GIF is animated: {str(e)}")
    
    # For SVG, consider it an illustration
    if mime_type == "image/svg+xml" or file_extension.lower() == ".svg":
        return "Illustrations"
    
    # For other image types, try to determine if it's a photo or illustration
    try:
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Simple heuristic: Count number of unique colors
        # Sample part of the image for efficiency
        width, height = img.size
        sample_size = min(100, min(width, height))
        resized_img = img.resize((sample_size, sample_size))
        pixels = list(resized_img.getdata())
        unique_colors = len(set(pixels))
        
        # If image has relatively few unique colors, likely an illustration
        if unique_colors < (sample_size * sample_size * 0.1):  # < 10% unique colors
            return "Illustrations"
        
        # Otherwise, consider it a photo
        return "Photos"
    except Exception as e:
        logging.warning(f"Error classifying image: {str(e)}")
        return "Photos"  # Default to Photos

# Function to determine file extension from URL
def get_file_extension(url):
    """Extract file extension from URL."""
    parsed_url = urlparse(url)
    path = parsed_url.path
    return os.path.splitext(path)[1]

# Function to generate image embedding using ResNet50
def generate_image_embedding(image_data):
    """Generate embedding for an image using ResNet50."""
    try:
        # Open image from binary data
        img = Image.open(io.BytesIO(image_data))
        # Convert to RGB if needed (handles PNG with alpha channel)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Preprocess image
        img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
        
        # Get embedding
        with torch.no_grad():
            embedding = image_embedding_model(img_tensor)
            # Flatten and convert to numpy array
            embedding = embedding.reshape(1, -1).numpy().astype(np.float32)
        
        return embedding
    except Exception as e:
        logging.error(f"Error generating image embedding: {str(e)}")
        return None

# Initialize FAISS index with database entries
async def initialize_faiss():
    await database.connect()
    
    # Initialize text search index
    query = "SELECT * FROM crawled_pages"
    pages = await database.fetch_all(query)

    for page in pages:
        embedding = embedding_model.encode([page["content"]])[0]
        faiss_index.add(np.array([embedding], dtype=np.float32))
        page_id_list.append(page["id"])
    
    logging.info(f"Initialized text FAISS index with {len(page_id_list)} pages")
    
    # Initialize image search index
    image_query = "SELECT * FROM crawled_images WHERE data IS NOT NULL"
    images = await database.fetch_all(image_query)
    
    for image in images:
        if not image["data"]:
            continue
            
        # Generate embedding for the image
        embedding = generate_image_embedding(image["data"])
        if embedding is not None:
            image_faiss_index.add(embedding)
            image_id = image["id"]
            image_id_list.append(image_id)
            
            # Get file extension from URL
            file_extension = get_file_extension(image["url"])
            
            # Classify image type
            image_type = classify_image_type(image["data"], file_extension, image["content_type"])
            
            # Store metadata
            try:
                timestamp = image["timestamp"]
                if not timestamp:  # Handle empty string or None
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            except (KeyError, AttributeError):
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
            image_metadata[image_id] = {
                "category": image_type,
                "timestamp": timestamp
            }
    
    logging.info(f"Initialized image FAISS index with {len(image_id_list)} images")

# Text search function
async def search_query(query: str, page: int = 1, results_per_page: int = 10) -> List[Dict[str, Any]]:
    results = []
    if query.strip():
        query_embedding = embedding_model.encode([query])[0]
        # Get a large number of results (e.g., 100) to paginate over
        _, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), 100)

        # Determine the slice of results to show based on the current page
        start_index = (page - 1) * results_per_page
        end_index = start_index + results_per_page
        paginated_indices = indices[0][start_index:end_index]

        # Retrieve the pages corresponding to the found indices
        for idx in paginated_indices:
            if idx < 0 or idx >= len(page_id_list):
                continue
            page_id = page_id_list[idx]
            query = f"SELECT * FROM crawled_pages WHERE id = {page_id}"
            page_result = await database.fetch_one(query)
            if page_result:
                # Include timestamp in results if available
                result = {
                    "title": page_result["title"],
                    "url": page_result["url"],
                    "snippet": page_result["snippet"]
                }
                
                # Add timestamp if it exists in the database
                if "timestamp" in page_result and page_result["timestamp"]:
                    result["timestamp"] = page_result["timestamp"]
                
                # Get associated images for this page (limit to 1 thumbnail per result)
                image_query = f"SELECT * FROM crawled_images WHERE page_url = :page_url LIMIT 1"
                image = await database.fetch_one(image_query, {"page_url": page_result["url"]})
                if image and image["data"]:
                    # Convert binary image data to base64 for embedding in HTML
                    image_b64 = base64.b64encode(image["data"]).decode("utf-8")
                    result["image"] = {
                        "data": f"data:{image['content_type']};base64,{image_b64}",
                        "alt_text": image["alt_text"] or "Page image",
                        "url": image["url"]
                    }
                
                results.append(result)
    return results

# Image search function with category filtering
async def search_images(query: str, page: int = 1, results_per_page: int = 20, category: str = "All") -> List[Dict[str, Any]]:
    """
    Search for images based on text query, with filtering by category.
    Categories: All, Photos, Illustrations, Animated, Recent
    """
    results = []
    if query.strip():
        query_embedding = embedding_model.encode([query])[0]
        # Get a larger number of results for image search
        _, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), 500)  # Get more to allow for filtering
        
        # Track unique images to avoid duplicates
        processed_image_urls = set()
        all_results = []  # Store all results before pagination
        
        # Process all matched pages to find relevant images
        for idx in indices[0]:
            if idx < 0 or idx >= len(page_id_list):
                continue
                
            page_id = page_id_list[idx]
            page_query = f"SELECT * FROM crawled_pages WHERE id = {page_id}"
            page_result = await database.fetch_one(page_query)
            
            if page_result:
                # Get all images from this page
                images_query = f"SELECT * FROM crawled_images WHERE page_url = :page_url"
                images = await database.fetch_all(images_query, {"page_url": page_result["url"]})
                
                for image in images:
                    # Skip already processed images
                    if image["url"] in processed_image_urls:
                        continue
                        
                    processed_image_urls.add(image["url"])
                    
                    # Skip images without binary data
                    if not image["data"]:
                        continue
                    
                    # Apply category filtering
                    image_id = image["id"]
                    meta = image_metadata.get(image_id, {})
                    image_category = meta.get("category", "Photos")  # Default to Photos if not classified
                    image_timestamp = meta.get("timestamp")
                    
                    # Filter by category if specified
                    if category != "All":
                        if category == "Recent":
                            # Check if image is recent (within the last 7 days)
                            if not image_timestamp:
                                continue
                                
                            try:
                                timestamp_dt = datetime.strptime(image_timestamp, '%Y-%m-%d %H:%M:%S')
                                if datetime.now() - timestamp_dt > timedelta(days=7):
                                    continue  # Skip if older than 7 days
                            except Exception:
                                continue  # Skip if timestamp can't be parsed
                        elif category != image_category:
                            continue  # Skip if doesn't match requested category
                    
                    # Convert binary image data to base64 for embedding in HTML
                    image_b64 = base64.b64encode(image["data"]).decode("utf-8")
                    
                    result = {
                        "image": {
                            "data": f"data:{image['content_type']};base64,{image_b64}",
                            "alt_text": image["alt_text"] or "Image",
                            "title": image["title"] or page_result["title"] or "Unnamed image",
                            "url": image["url"],
                            "category": image_category,
                            "timestamp": image_timestamp,
                            "source_page": {
                                "title": page_result["title"],
                                "url": page_result["url"]
                            }
                        }
                    }
                    
                    all_results.append(result)
        
        # Apply pagination after filtering
        start_index = (page - 1) * results_per_page
        end_index = start_index + results_per_page
        results = all_results[start_index:end_index] if start_index < len(all_results) else []
        
    return results

# New function for reverse image search with category filtering
async def search_by_image(image_data, page: int = 1, results_per_page: int = 20, category: str = "All") -> List[Dict[str, Any]]:
    """Search for similar images based on an uploaded image with category filtering."""
    results = []
    
    # Generate embedding for the uploaded image
    query_embedding = generate_image_embedding(image_data)
    
    if query_embedding is not None and len(image_id_list) > 0:
        # Search for similar images in the image index
        distances, indices = image_faiss_index.search(query_embedding, 500)  # Get more to allow for filtering
        
        all_results = []  # Store all results before pagination
        
        # Retrieve the images corresponding to the found indices
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(image_id_list):
                continue
                
            image_id = image_id_list[idx]
            
            # Apply category filtering
            meta = image_metadata.get(image_id, {})
            image_category = meta.get("category", "Photos")  # Default to Photos if not classified
            image_timestamp = meta.get("timestamp")
            
            # Filter by category if specified
            if category != "All":
                if category == "Recent":
                    # Check if image is recent (within the last 7 days)
                    if not image_timestamp:
                        continue
                        
                    try:
                        timestamp_dt = datetime.strptime(image_timestamp, '%Y-%m-%d %H:%M:%S')
                        if datetime.now() - timestamp_dt > timedelta(days=7):
                            continue  # Skip if older than 7 days
                    except Exception:
                        continue  # Skip if timestamp can't be parsed
                elif category != image_category:
                    continue  # Skip if doesn't match requested category
                    
            image_query = f"SELECT * FROM crawled_images WHERE id = {image_id}"
            image_result = await database.fetch_one(image_query)
            
            if image_result and image_result["data"]:
                # Get the associated page
                page_query = f"SELECT * FROM crawled_pages WHERE url = :page_url"
                page_result = await database.fetch_one(page_query, {"page_url": image_result["page_url"]})
                
                # Convert binary image data to base64 for embedding in HTML
                image_b64 = base64.b64encode(image_result["data"]).decode("utf-8")
                
                result = {
                    "image": {
                        "data": f"data:{image_result['content_type']};base64,{image_b64}",
                        "alt_text": image_result["alt_text"] or "Image",
                        "title": image_result["title"] or (page_result["title"] if page_result else "") or "Unnamed image",
                        "url": image_result["url"],
                        "category": image_category,
                        "timestamp": image_timestamp,
                        "similarity_score": float(100 - distances[0][i]),  # Convert distance to similarity score
                        "source_page": {
                            "title": page_result["title"] if page_result else "Unknown page",
                            "url": page_result["url"] if page_result else "#"
                        }
                    }
                }
                
                all_results.append(result)
        
        # Apply pagination after filtering
        start_index = (page - 1) * results_per_page
        end_index = start_index + results_per_page
        results = all_results[start_index:end_index] if start_index < len(all_results) else []
    
    return results

# Simple HTTP handler for serving search results and homepage
class SearchRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)

        if parsed_path.path == "/":
            # Serve the homepage with the search bar
            html_content = index_template.render()
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))

        elif parsed_path.path == "/search":
            # Handle search query and serve results page
            query = query_params.get("q", [""])[0]
            page = int(query_params.get("page", [1])[0])  # Get page number from query params (default to 1)
            search_type = query_params.get("type", ["web"])[0]  # Get search type (web or images)
            category = query_params.get("category", ["All"])[0]  # Get image category filter
            
            if search_type == "images":
                # Handle image search with category filtering
                results = asyncio.run(search_images(query, page, results_per_page=20, category=category))
                
                # Count total results for pagination (this would normally be calculated dynamically)
                # Here we'll just use an estimate based on the category
                total_results = 100
                if category != "All":
                    total_results = 50  # Fewer results when filtering
                
                total_pages = (total_results // 20) + (1 if total_results % 20 > 0 else 0)
                
                html_content = image_results_template.render(
                    query=query,
                    results=results,
                    current_page=page,
                    total_pages=total_pages,
                    selected_category=category
                )
            else:
                # Handle web search (default)
                results = asyncio.run(search_query(query, page, results_per_page=10))
                total_results = 100  # This should be dynamically calculated
                total_pages = (total_results // 10) + (1 if total_results % 10 > 0 else 0)
                
                html_content = results_template.render(
                    query=query,
                    results=results,
                    current_page=page,
                    total_pages=total_pages
                )

            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))
            
        elif parsed_path.path == "/image":
            # Handle direct image access by ID
            image_id = query_params.get("id", [""])[0]
            if image_id:
                try:
                    # Fetch the image from the database
                    image_query = f"SELECT * FROM crawled_images WHERE id = :image_id"
                    image = asyncio.run(database.fetch_one(image_query, {"image_id": int(image_id)}))
                    
                    if image and image["data"]:
                        self.send_response(200)
                        self.send_header("Content-type", image["content_type"] or "image/jpeg")
                        self.end_headers()
                        self.wfile.write(image["data"])
                        return
                except Exception as e:
                    logging.error(f"Error serving image {image_id}: {str(e)}")
            
            # If we get here, the image wasn't found or there was an error
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Image not found")
            
        elif parsed_path.path == "/image-search":
            # Serve the image search upload form
            html_content = image_search_template.render()
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))

        else:
            super().do_GET()
    
    def do_POST(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == "/image-search":
            # Handle image upload for reverse image search
            content_type, pdict = cgi.parse_header(self.headers.get('Content-Type', ''))
            
            if content_type == 'multipart/form-data':
                # Parse the multipart form data
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST'}
                )
                
                # Check if the file was uploaded
                if 'image' in form and form['image'].file:
                    # Read the uploaded file
                    image_data = form['image'].file.read()
                    
                    # Ensure image_data is bytes, not string
                    if isinstance(image_data, str):
                        image_data = image_data.encode('utf-8')
                    
                    # Save the uploaded file with a unique name
                    filename = str(uuid.uuid4()) + ".jpg"
                    filepath = os.path.join(UPLOAD_DIR, filename)
                    with open(filepath, 'wb') as f:
                        f.write(image_data)
                    
                    # Get category filter if provided
                    category = form.getvalue('category', 'All')
                    
                    # Perform reverse image search with category filtering
                    page = int(form.getvalue('page', 1))
                    results = asyncio.run(search_by_image(image_data, page, results_per_page=20, category=category))
                    
                    # Calculate pagination info
                    total_results = 100  # This should be dynamically calculated
                    if category != "All":
                        total_results = 50  # Fewer results when filtering
                        
                    total_pages = (total_results // 20) + (1 if total_results % 20 > 0 else 0)
                    
                    # Create a base64 version of the uploaded image for display
                    uploaded_image_b64 = base64.b64encode(image_data).decode("utf-8")
                    
                    # Render the results page
                    html_content = image_search_results_template.render(
                        uploaded_image=f"data:image/jpeg;base64,{uploaded_image_b64}",
                        results=results,
                        current_page=page,
                        total_pages=total_pages,
                        selected_category=category
                    )
                    
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(html_content.encode("utf-8"))
                    return
            
            # If we get here, there was an error with the image upload
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Error: Missing or invalid image file</h1><p>Please <a href='/image-search'>try again</a> with a valid image file.</p></body></html>")
        else:
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not found")

# Main function to start the server and initialize FAISS
def run_server():
    # Initialize FAISS with database entries before starting the server
    asyncio.run(initialize_faiss())
    
    # Start HTTP server
    server_address = ("", 8000)
    httpd = HTTPServer(server_address, SearchRequestHandler)
    logging.info("Starting HTTP server at http://localhost:8000")
    httpd.serve_forever()

if __name__ == "__main__":
    nest_asyncio.apply()
    logging.basicConfig(level=logging.INFO)
    run_server()
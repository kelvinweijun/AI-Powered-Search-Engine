## Search Engine and Web Crawler

![Logo](/static/logo.png)

This project combines a **depth-first web crawler** with an **AI-powered search engine with reverse image search capabilities**. It allows users to index the contents of web pages including images, into a relational database and later query that data using an image, returning similar images and associated page content.

---

## Features

### Web Crawler (`web_crawler.py`)
- Implements a **tree-based, depth-first traversal**.
- Uses `BeautifulSoup` for HTML parsing and scraping.
- Extracts:
  - Text snippets from each webpage.
  - All downloadable images.
- Stores data in a **MariaDB** database using **SQLAlchemy ORM**.
- Avoids duplicate indexing and revisiting previously crawled links.

### Search Engine (`search_engine.py`)
- Performs both **text based search** and **reverse image search**:
  - Encodes images with **ResNet-50** (via Torchvision).
  - Uses **FAISS** for high-performance similarity search over text vectors.
- Returns the most visually similar images from the indexed set.
- Provides metadata like the source webpage and content snippet for each match.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/reverse-image-search-crawler.git
cd reverse-image-search-crawler
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up MariaDB
Make sure you've installed MariaDB and a suitable database management platform (preferable HeidiSQL or Table Plus). 
Set up a database connection with the following credentials:

host: 127.0.0.1

user: root

password: root

Then, create a new database named "search_engine"

### 4. Set seed link in `web_crawler.py`

Open web_crawler.py and change the seed link to whatever you want.

### 5. Run `web_crawler.py`

Run web_crawler.py to begin crawling the pages. 

### 6. Run `search_engine.py`

Run search_engine.py to start the search engine. This might take a while because the indexes are being embedded. Once done, access the localhost website on your browser and you should be able to see the webpage.


## Search Engine and Web Crawler

This project combines a **depth-first web crawler** with a **reverse image search engine**. It allows users to index the contents of web pages—including images—into a relational database and later query that data using an image, returning similar images and associated page content.

---

## 🛠️ Features

### 🌐 Web Crawler (`crawler.py`)
- Implements a **tree-based, depth-first traversal**.
- Uses `BeautifulSoup` for HTML parsing and scraping.
- Extracts:
  - Text snippets from each webpage.
  - All downloadable images.
- Stores data in a **MariaDB** database using **SQLAlchemy ORM**.
- Avoids duplicate indexing and revisiting previously crawled links.

### 🔍 Search Engine (`search_engine.py`)
- Performs **reverse image search**:
  - Encodes images with **ResNet-50** (via Torchvision).
  - Uses **FAISS** for high-performance similarity search over image vectors.
- Returns the most visually similar images from the indexed set.
- Provides metadata like the source webpage and content snippet for each match.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/reverse-image-search-crawler.git
cd reverse-image-search-crawler

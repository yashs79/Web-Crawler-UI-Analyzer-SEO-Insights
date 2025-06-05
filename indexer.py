#!/usr/bin/env python3
"""
Indexer component that processes and stores crawled web pages.
"""
import os
import json
import sqlite3
import logging
import pandas as pd
import networkx as nx
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class Indexer:
    """Indexes and stores crawled web pages for analysis."""
    
    def __init__(self, db_path='seo_index.db'):
        """
        Initialize the indexer with a database path.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.link_graph = nx.DiGraph()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize the SQLite database with required tables."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Pages table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY,
            url TEXT UNIQUE,
            title TEXT,
            meta_description TEXT,
            meta_keywords TEXT,
            content_length INTEGER,
            h1_count INTEGER,
            h2_count INTEGER,
            img_count INTEGER,
            has_structured_data BOOLEAN,
            crawl_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Links table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS links (
            id INTEGER PRIMARY KEY,
            source_url TEXT,
            target_url TEXT,
            anchor_text TEXT,
            is_internal BOOLEAN,
            FOREIGN KEY (source_url) REFERENCES pages (url),
            UNIQUE (source_url, target_url)
        )
        ''')
        
        # Keywords table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS keywords (
            id INTEGER PRIMARY KEY,
            url TEXT,
            keyword TEXT,
            frequency INTEGER,
            position_score REAL,
            FOREIGN KEY (url) REFERENCES pages (url),
            UNIQUE (url, keyword)
        )
        ''')
        
        # SEO scores table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS seo_scores (
            id INTEGER PRIMARY KEY,
            url TEXT UNIQUE,
            content_quality_score REAL,
            backlink_authority_score REAL,
            technical_seo_score REAL,
            user_experience_score REAL,
            search_intent_score REAL,
            page_speed_score REAL,
            brand_social_score REAL,
            total_seo_score REAL,
            analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (url) REFERENCES pages (url)
        )
        ''')
        
        self.conn.commit()
        
    def _extract_text_content(self, html):
        """Extract and clean text content from HTML."""
        soup = BeautifulSoup(html, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
        
    def _extract_keywords(self, text, max_keywords=100):
        """Extract keywords from text using NLP techniques."""
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        filtered_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token.isalpha() and token not in self.stop_words and len(token) > 2
        ]
        
        # Count frequencies
        keyword_freq = Counter(filtered_tokens)
        
        # Return top keywords
        return keyword_freq.most_common(max_keywords)
    
    def _extract_links(self, url, html):
        """Extract links from HTML content."""
        soup = BeautifulSoup(html, 'lxml')
        base_domain = url.split('//')[1].split('/')[0]
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            anchor_text = a_tag.get_text().strip()
            
            # Skip empty or javascript links
            if not href or href.startswith(('javascript:', '#')):
                continue
                
            # Convert relative URLs to absolute
            if href.startswith('/'):
                scheme = url.split('//')[0]
                href = f"{scheme}//{base_domain}{href}"
            elif not href.startswith(('http://', 'https://')):
                if url.endswith('/'):
                    href = f"{url}{href}"
                else:
                    href = f"{url}/{href}"
            
            # Determine if internal link
            is_internal = base_domain in href
            
            links.append({
                'source_url': url,
                'target_url': href,
                'anchor_text': anchor_text,
                'is_internal': is_internal
            })
            
        return links
    
    def index_page(self, page_data):
        """
        Index a single page and store its data.
        
        Args:
            page_data: Dictionary containing page data from crawler
        """
        url = page_data['url']
        html = page_data['html']
        
        # Extract text content
        text_content = self._extract_text_content(html)
        
        # Store page data
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT OR REPLACE INTO pages 
        (url, title, meta_description, meta_keywords, content_length, 
         h1_count, h2_count, img_count, has_structured_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            url,
            page_data['title'],
            page_data['meta_description'],
            page_data['meta_keywords'],
            len(text_content),
            len(page_data['h1_tags']),
            len(page_data['h2_tags']),
            len(page_data['images']),
            bool(page_data['structured_data'])
        ))
        
        # Extract and store keywords
        keywords = self._extract_keywords(text_content)
        for keyword, frequency in keywords:
            # Calculate position score (higher if keyword appears in title, headings, etc.)
            position_score = 1.0
            if keyword in page_data['title'].lower():
                position_score += 3.0
            if any(keyword in h.lower() for h in page_data['h1_tags']):
                position_score += 2.0
            if any(keyword in h.lower() for h in page_data['h2_tags']):
                position_score += 1.0
            if keyword in page_data['meta_description'].lower():
                position_score += 1.5
                
            cursor.execute('''
            INSERT OR REPLACE INTO keywords
            (url, keyword, frequency, position_score)
            VALUES (?, ?, ?, ?)
            ''', (url, keyword, frequency, position_score))
        
        # Extract and store links
        links = self._extract_links(url, html)
        for link in links:
            cursor.execute('''
            INSERT OR IGNORE INTO links
            (source_url, target_url, anchor_text, is_internal)
            VALUES (?, ?, ?, ?)
            ''', (
                link['source_url'],
                link['target_url'],
                link['anchor_text'],
                link['is_internal']
            ))
            
            # Add to link graph
            self.link_graph.add_edge(
                link['source_url'],
                link['target_url'],
                anchor=link['anchor_text']
            )
        
        self.conn.commit()
    
    def index_crawl_results(self, crawl_results):
        """
        Index all pages from crawl results.
        
        Args:
            crawl_results: Dictionary of crawled pages from crawler
        """
        for url, page_data in crawl_results.items():
            self.index_page(page_data)
            
        logger.info(f"Indexed {len(crawl_results)} pages")
        
    def get_page_data(self, url):
        """Get indexed data for a specific page."""
        cursor = self.conn.cursor()
        
        # Get page data
        cursor.execute('SELECT * FROM pages WHERE url = ?', (url,))
        page_data = cursor.fetchone()
        
        if not page_data:
            return None
            
        # Get keywords
        cursor.execute('SELECT keyword, frequency, position_score FROM keywords WHERE url = ?', (url,))
        keywords = cursor.fetchall()
        
        # Get incoming links
        cursor.execute('SELECT source_url, anchor_text FROM links WHERE target_url = ?', (url,))
        incoming_links = cursor.fetchall()
        
        # Get outgoing links
        cursor.execute('SELECT target_url, anchor_text FROM links WHERE source_url = ?', (url,))
        outgoing_links = cursor.fetchall()
        
        return {
            'page_data': page_data,
            'keywords': keywords,
            'incoming_links': incoming_links,
            'outgoing_links': outgoing_links
        }
        
    def get_link_graph(self):
        """Get the link graph for visualization and analysis."""
        return self.link_graph
        
    def calculate_pagerank(self):
        """Calculate PageRank for all pages in the link graph."""
        pagerank = nx.pagerank(self.link_graph)
        return pagerank
        
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

# Command line interface for testing
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python indexer.py <crawl_results_json>")
        sys.exit(1)
        
    # Load crawl results from JSON file
    with open(sys.argv[1], 'r') as f:
        crawl_results = json.load(f)
        
    # Index the results
    indexer = Indexer()
    indexer.index_crawl_results(crawl_results)
    
    # Calculate PageRank
    pagerank = indexer.calculate_pagerank()
    print("PageRank scores:")
    for url, score in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{url}: {score:.4f}")
        
    indexer.close()

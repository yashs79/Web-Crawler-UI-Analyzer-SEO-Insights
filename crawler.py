#!/usr/bin/env python3
"""
Web crawler component that fetches and processes web pages.
"""
import time
import re
import urllib.parse
from urllib.robotparser import RobotFileParser
from collections import deque
import logging
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Crawler:
    """Google-like web crawler that respects robots.txt and crawl delays."""
    
    def __init__(self, max_depth=3, max_pages=100, respect_robots=True, 
                 crawl_delay=1, allowed_domains=None, user_agent="SEOAnalyzerBot/1.0"):
        """
        Initialize the crawler with configuration parameters.
        
        Args:
            max_depth: Maximum link depth to crawl
            max_pages: Maximum number of pages to crawl
            respect_robots: Whether to respect robots.txt rules
            crawl_delay: Delay between requests to the same domain (seconds)
            allowed_domains: List of domains to restrict crawling to
            user_agent: User agent string to identify the crawler
        """
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.respect_robots = respect_robots
        self.crawl_delay = crawl_delay
        self.allowed_domains = allowed_domains or []
        self.user_agent = user_agent
        
        # Tracking state
        self.visited_urls = set()
        self.urls_to_visit = deque()
        self.domain_last_crawled = {}
        self.robot_parsers = {}
        self.crawl_results = {}
        
    def _get_domain(self, url):
        """Extract domain from URL."""
        parsed = urllib.parse.urlparse(url)
        return parsed.netloc
    
    def _is_allowed_domain(self, url):
        """Check if URL's domain is in allowed domains list."""
        if not self.allowed_domains:
            return True
        return self._get_domain(url) in self.allowed_domains
    
    def _is_allowed_by_robots(self, url):
        """Check if URL is allowed by robots.txt."""
        if not self.respect_robots:
            return True
            
        domain = self._get_domain(url)
        if domain not in self.robot_parsers:
            robots_url = f"{urllib.parse.urlparse(url).scheme}://{domain}/robots.txt"
            parser = RobotFileParser()
            parser.set_url(robots_url)
            try:
                parser.read()
                self.robot_parsers[domain] = parser
            except Exception as e:
                logger.warning(f"Error reading robots.txt for {domain}: {e}")
                self.robot_parsers[domain] = None
                
        parser = self.robot_parsers[domain]
        if parser:
            return parser.can_fetch(self.user_agent, url)
        return True
    
    def _respect_crawl_delay(self, domain):
        """Respect crawl delay for a domain."""
        if domain in self.domain_last_crawled:
            last_crawl = self.domain_last_crawled[domain]
            elapsed = time.time() - last_crawl
            if elapsed < self.crawl_delay:
                time.sleep(self.crawl_delay - elapsed)
        
        self.domain_last_crawled[domain] = time.time()
    
    def _normalize_url(self, url, base_url):
        """Normalize relative URLs to absolute URLs."""
        return urllib.parse.urljoin(base_url, url)
    
    def _is_valid_url(self, url):
        """Check if URL is valid and should be crawled."""
        # Skip non-HTTP URLs
        if not url.startswith(('http://', 'https://')):
            return False
            
        # Skip URLs with fragments
        if '#' in url:
            url = url.split('#')[0]
            
        # Skip common non-HTML resources
        if re.search(r'\.(jpg|jpeg|png|gif|pdf|zip|tar|gz|mp3|mp4|avi|mov|wmv|flv|css|js)$', url, re.IGNORECASE):
            return False
            
        return True
    
    async def _fetch_url(self, url, session):
        """Fetch URL content using aiohttp."""
        try:
            domain = self._get_domain(url)
            self._respect_crawl_delay(domain)
            
            headers = {'User-Agent': self.user_agent}
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' in content_type:
                        return await response.text()
                    else:
                        logger.info(f"Skipping non-HTML content: {url} ({content_type})")
                        return None
                else:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def _extract_links(self, html, base_url):
        """Extract links from HTML content."""
        soup = BeautifulSoup(html, 'lxml')
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = self._normalize_url(href, base_url)
            
            if self._is_valid_url(full_url):
                links.append(full_url)
                
        return links
    
    def _extract_page_data(self, url, html):
        """Extract relevant data from a page for SEO analysis."""
        soup = BeautifulSoup(html, 'lxml')
        
        # Basic page data
        title = soup.title.string if soup.title else ""
        meta_description = ""
        meta_keywords = ""
        
        # Get meta tags
        for meta in soup.find_all('meta'):
            if meta.get('name', '').lower() == 'description':
                meta_description = meta.get('content', '')
            elif meta.get('name', '').lower() == 'keywords':
                meta_keywords = meta.get('content', '')
        
        # Get headings
        h1_tags = [h.get_text().strip() for h in soup.find_all('h1')]
        h2_tags = [h.get_text().strip() for h in soup.find_all('h2')]
        
        # Get main content (simplified)
        main_content = ""
        for tag in soup.find_all(['p', 'article', 'section', 'main']):
            main_content += tag.get_text().strip() + " "
        
        # Get images with alt text
        images = []
        for img in soup.find_all('img'):
            images.append({
                'src': img.get('src', ''),
                'alt': img.get('alt', '')
            })
        
        # Get structured data
        structured_data = []
        for script in soup.find_all('script', {'type': 'application/ld+json'}):
            structured_data.append(script.string)
        
        return {
            'url': url,
            'title': title,
            'meta_description': meta_description,
            'meta_keywords': meta_keywords,
            'h1_tags': h1_tags,
            'h2_tags': h2_tags,
            'content': main_content,
            'images': images,
            'structured_data': structured_data,
            'html': html,  # Store full HTML for further analysis
        }
    
    async def crawl(self, start_url):
        """
        Start crawling from the given URL.
        
        Args:
            start_url: The URL to start crawling from
            
        Returns:
            Dictionary of crawled pages with their data
        """
        if not self._is_valid_url(start_url):
            logger.error(f"Invalid start URL: {start_url}")
            return {}
            
        # Reset state
        self.visited_urls = set()
        self.urls_to_visit = deque([(start_url, 0)])  # (url, depth)
        self.crawl_results = {}
        
        # Create progress bar
        progress = tqdm(total=self.max_pages, desc="Crawling")
        
        # Create aiohttp session
        async with aiohttp.ClientSession() as session:
            while self.urls_to_visit and len(self.visited_urls) < self.max_pages:
                url, depth = self.urls_to_visit.popleft()
                
                if url in self.visited_urls:
                    continue
                    
                if not self._is_allowed_domain(url) or not self._is_allowed_by_robots(url):
                    continue
                
                # Fetch page
                html = await self._fetch_url(url, session)
                if html:
                    # Process page
                    self.visited_urls.add(url)
                    page_data = self._extract_page_data(url, html)
                    self.crawl_results[url] = page_data
                    progress.update(1)
                    
                    # Extract links if not at max depth
                    if depth < self.max_depth:
                        links = self._extract_links(html, url)
                        for link in links:
                            if link not in self.visited_urls:
                                self.urls_to_visit.append((link, depth + 1))
        
        progress.close()
        logger.info(f"Crawl completed. Visited {len(self.visited_urls)} pages.")
        return self.crawl_results

    def crawl_url(self, url):
        """
        Non-async wrapper for the crawl method to make it easier to use.
        
        Args:
            url: URL to crawl
            
        Returns:
            Dictionary of crawled pages with their data
        """
        return asyncio.run(self.crawl(url))

# Command line interface for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python crawler.py <start_url>")
        sys.exit(1)
        
    start_url = sys.argv[1]
    crawler = Crawler(max_depth=2, max_pages=10)
    
    # Run the crawler
    results = crawler.crawl_url(start_url)
    
    # Print results
    for url, data in results.items():
        print(f"URL: {url}")
        print(f"Title: {data['title']}")
        print(f"Meta Description: {data['meta_description'][:100]}...")
        print("-" * 80)

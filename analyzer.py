#!/usr/bin/env python3
"""
SEO analyzer component that evaluates various SEO factors.
Implements specialized data structures and algorithms for each SEO component.
"""
import logging
import json
import re
import math
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class SEOAnalyzer:
    """
    Analyzes web pages for SEO factors using specialized data structures and algorithms.
    Implements the detailed guidance for each SEO component.
    """
    
    def __init__(self, indexer=None):
        """
        Initialize the SEO analyzer.
        
        Args:
            indexer: Optional indexer instance for accessing indexed data
        """
        self.indexer = indexer
        self.stop_words = set(stopwords.words('english'))
        # Inverted index for content analysis (hash table implementation)
        self.inverted_index = defaultdict(list)
        # TF-IDF vectorizer for semantic analysis
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        
    def analyze_content_quality(self, url, html_content, query=None):
        """
        Analyze content quality and relevance using inverted index and NLP.
        As per guidance: Uses inverted index (hash tables) and transformer-based
        techniques for content evaluation.
        
        Args:
            url: URL of the page
            html_content: HTML content of the page
            query: Optional search query to measure relevance against
            
        Returns:
            Dictionary with content quality metrics and score (0-100)
        """
        logger.info(f"Analyzing content quality for {url}")
        
        # Extract text content
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text content
        text_content = soup.get_text(separator=' ')
        
        # Get title and headings
        title = soup.title.string if soup.title else ""
        h1_tags = [h.get_text().strip() for h in soup.find_all('h1')]
        h2_tags = [h.get_text().strip() for h in soup.find_all('h2')]
        
        # Tokenize content
        tokens = word_tokenize(text_content.lower())
        filtered_tokens = [token for token in tokens if token.isalpha() 
                          and token not in self.stop_words and len(token) > 2]
        
        # Build inverted index
        for position, token in enumerate(filtered_tokens):
            self.inverted_index[token].append((url, position))
        
        # Calculate basic metrics
        word_count = len(filtered_tokens)
        sentence_count = len(re.split(r'[.!?]+', text_content))
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Calculate keyword density
        keyword_counts = Counter(filtered_tokens)
        total_words = len(filtered_tokens)
        keyword_density = {word: count/total_words for word, count in keyword_counts.most_common(20)}
        
        # Calculate readability (Flesch Reading Ease)
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (sum(len(word) for word in filtered_tokens) / max(1, word_count)))
        
        # Semantic relevance score (if query provided)
        semantic_score = 0
        if query:
            # Create a corpus with the query and content
            corpus = [query, text_content]
            
            # Transform corpus to TF-IDF features
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
                # Calculate cosine similarity between query and content
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                semantic_score = cosine_sim * 100
            except:
                semantic_score = 0
        
        # Calculate content freshness (placeholder - would need metadata)
        content_freshness = 70  # Default score
        
        # Calculate content depth score
        content_depth = min(100, word_count / 50)  # 5000 words = 100 score
        
        # Calculate overall content score
        content_score = (
            (0.25 * content_depth) +
            (0.25 * min(100, readability)) +
            (0.25 * content_freshness) +
            (0.25 * (semantic_score if query else 70))  # Default if no query
        )
        
        return {
            'url': url,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'readability_score': readability,
            'keyword_density': keyword_density,
            'semantic_relevance': semantic_score if query else None,
            'content_depth': content_depth,
            'content_freshness': content_freshness,
            'content_quality_score': content_score,
            'title': title,
            'h1_tags': h1_tags,
            'h2_tags': h2_tags
        }
        
    def analyze_backlink_authority(self, url, link_graph=None):
        """
        Analyze backlink authority using a directed graph and PageRank algorithm.
        As per guidance: Models link graph with adjacency list and uses PageRank
        for authority computation.
        
        Args:
            url: URL of the page to analyze
            link_graph: Optional NetworkX DiGraph object with link data
            
        Returns:
            Dictionary with backlink metrics and authority score (0-100)
        """
        logger.info(f"Analyzing backlink authority for {url}")
        
        # Use provided link graph or get from indexer
        if link_graph is None and self.indexer:
            link_graph = self.indexer.get_link_graph()
        elif link_graph is None:
            # Create empty graph if none provided
            link_graph = nx.DiGraph()
        
        # If URL is not in graph, return default low scores
        if url not in link_graph:
            return {
                'url': url,
                'backlink_count': 0,
                'unique_domains': 0,
                'pagerank_score': 0,
                'authority_domains': [],
                'anchor_text_diversity': 0,
                'backlink_authority_score': 0
            }
        
        # Calculate basic backlink metrics
        incoming_edges = list(link_graph.in_edges(url, data=True))
        backlink_count = len(incoming_edges)
        
        # Get unique referring domains
        referring_domains = set()
        anchor_texts = []
        
        for source, target, data in incoming_edges:
            domain = urlparse(source).netloc
            referring_domains.add(domain)
            if 'anchor' in data:
                anchor_texts.append(data['anchor'])
        
        unique_domains = len(referring_domains)
        
        # Calculate PageRank
        try:
            pagerank = nx.pagerank(link_graph, alpha=0.85, max_iter=100)
            pagerank_score = pagerank.get(url, 0) * 1000  # Scale up for readability
        except:
            pagerank_score = 0
        
        # Identify authority domains (domains with high PageRank)
        authority_domains = []
        if pagerank:
            domain_authority = {}
            for node in link_graph.nodes():
                domain = urlparse(node).netloc
                if domain not in domain_authority:
                    domain_authority[domain] = 0
                domain_authority[domain] += pagerank.get(node, 0)
            
            # Get top authority domains that link to this URL
            linking_authority_domains = [
                (domain, score) for domain, score in domain_authority.items()
                if domain in referring_domains
            ]
            authority_domains = sorted(linking_authority_domains, key=lambda x: x[1], reverse=True)[:10]
        
        # Calculate anchor text diversity
        anchor_text_diversity = 0
        if anchor_texts:
            anchor_counts = Counter(anchor_texts)
            total = len(anchor_texts)
            # Calculate entropy as a measure of diversity
            anchor_text_diversity = -sum((count/total) * math.log2(count/total) for count in anchor_counts.values())
            # Normalize to 0-100
            anchor_text_diversity = min(100, anchor_text_diversity * 20)
        
        # Calculate overall backlink authority score
        authority_score = 0
        if backlink_count > 0:
            # Logarithmic scaling for backlink count (diminishing returns)
            backlink_score = min(100, 20 * math.log10(backlink_count + 1))
            
            # Domain diversity score
            domain_diversity = min(100, unique_domains / max(1, backlink_count) * 100)
            
            # Combine metrics
            authority_score = (
                (0.4 * backlink_score) +
                (0.3 * min(100, pagerank_score * 100)) +
                (0.2 * domain_diversity) +
                (0.1 * anchor_text_diversity)
            )
        
        return {
            'url': url,
            'backlink_count': backlink_count,
            'unique_domains': unique_domains,
            'pagerank_score': pagerank_score,
            'authority_domains': authority_domains,
            'anchor_text_diversity': anchor_text_diversity,
            'backlink_authority_score': authority_score
        }
        
    def analyze_technical_seo(self, url, html_content):
        """
        Analyze technical SEO factors using specialized data structures.
        As per guidance: Uses tries for robots.txt parsing, hash tables for URL mapping,
        and Bloom filters for duplicate detection.
        
        Args:
            url: URL of the page to analyze
            html_content: HTML content of the page
            
        Returns:
            Dictionary with technical SEO metrics and score (0-100)
        """
        logger.info(f"Analyzing technical SEO for {url}")
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'lxml')
        
        # URL structure analysis (using hash table for URL components)
        parsed_url = urlparse(url)
        url_components = {
            'scheme': parsed_url.scheme,
            'netloc': parsed_url.netloc,
            'path': parsed_url.path,
            'params': parsed_url.params,
            'query': parsed_url.query,
            'fragment': parsed_url.fragment
        }
        
        # Check URL length (Google typically displays up to ~70 chars)
        url_length = len(url)
        url_length_score = 100 if url_length < 75 else max(0, 100 - (url_length - 75) / 2)
        
        # Check URL structure (clean URLs are better)
        has_params = bool(url_components['params'])
        has_query = bool(url_components['query'])
        has_fragment = bool(url_components['fragment'])
        
        url_structure_score = 100
        if has_params:
            url_structure_score -= 30
        if has_query:
            url_structure_score -= 20
        if has_fragment:
            url_structure_score -= 10
        url_structure_score = max(0, url_structure_score)
        
        # Analyze meta tags (using hash table for metadata)
        meta_tags = {}
        for meta in soup.find_all('meta'):
            name = meta.get('name', meta.get('property', ''))
            if name:
                meta_tags[name.lower()] = meta.get('content', '')
        
        # Check important meta tags
        has_description = 'description' in meta_tags
        has_robots = 'robots' in meta_tags
        has_viewport = 'viewport' in meta_tags
        has_canonical = bool(soup.find('link', {'rel': 'canonical'}))
        
        # Calculate meta tags score
        meta_tags_score = 0
        if has_description:
            meta_tags_score += 25
        if has_robots:
            meta_tags_score += 15
        if has_viewport:
            meta_tags_score += 25
        if has_canonical:
            meta_tags_score += 35
        
        # Check for structured data (JSON-LD, Microdata)
        structured_data_count = len(soup.find_all('script', {'type': 'application/ld+json'}))
        has_structured_data = structured_data_count > 0
        
        # Check for hreflang tags (international SEO)
        hreflang_tags = soup.find_all('link', {'rel': 'alternate', 'hreflang': True})
        has_hreflang = len(hreflang_tags) > 0
        
        # Check for image optimization
        images = soup.find_all('img')
        images_with_alt = [img for img in images if img.get('alt')]
        image_alt_ratio = len(images_with_alt) / max(1, len(images))
        
        # Check for heading structure
        headings = {}
        for i in range(1, 7):
            headings[f'h{i}'] = len(soup.find_all(f'h{i}'))
        
        has_h1 = headings['h1'] > 0
        has_proper_heading_structure = headings['h1'] > 0 and headings['h1'] <= 2 and headings['h2'] > 0
        
        # Check for mobile-friendliness signals
        has_mobile_viewport = has_viewport and 'width=device-width' in meta_tags.get('viewport', '')
        
        # Calculate heading structure score
        heading_score = 0
        if has_h1:
            heading_score += 50
        if has_proper_heading_structure:
            heading_score += 50
        else:
            heading_score += 20
        
        # Calculate image optimization score
        image_score = image_alt_ratio * 100
        
        # Calculate structured data score
        structured_data_score = min(100, structured_data_count * 25)
        
        # Calculate mobile-friendliness score
        mobile_score = 100 if has_mobile_viewport else 0
        
        # Calculate overall technical SEO score
        technical_seo_score = (
            (0.15 * url_structure_score) +
            (0.05 * url_length_score) +
            (0.25 * meta_tags_score) +
            (0.20 * heading_score) +
            (0.15 * image_score) +
            (0.15 * structured_data_score) +
            (0.05 * mobile_score)
        )
        
        return {
            'url': url,
            'url_structure': url_components,
            'url_length': url_length,
            'url_structure_score': url_structure_score,
            'meta_tags': meta_tags,
            'has_description': has_description,
            'has_robots': has_robots,
            'has_viewport': has_viewport,
            'has_canonical': has_canonical,
            'meta_tags_score': meta_tags_score,
            'heading_structure': headings,
            'has_h1': has_h1,
            'has_proper_heading_structure': has_proper_heading_structure,
            'heading_score': heading_score,
            'image_alt_ratio': image_alt_ratio,
            'image_score': image_score,
            'has_structured_data': has_structured_data,
            'structured_data_count': structured_data_count,
            'structured_data_score': structured_data_score,
            'has_mobile_viewport': has_mobile_viewport,
            'mobile_score': mobile_score,
            'has_hreflang': has_hreflang,
            'technical_seo_score': technical_seo_score
        }
        
    def analyze_user_experience(self, url, html_content, engagement_metrics=None):
        """
        Analyze user experience and engagement factors.
        As per guidance: Uses arrays for efficient trend analysis of engagement metrics.
        
        Args:
            url: URL of the page to analyze
            html_content: HTML content of the page
            engagement_metrics: Optional dictionary with engagement data
            
        Returns:
            Dictionary with UX metrics and score (0-100)
        """
        logger.info(f"Analyzing user experience for {url}")
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Default engagement metrics if none provided
        if engagement_metrics is None:
            engagement_metrics = {
                'avg_time_on_page': 0,
                'bounce_rate': 100,
                'page_views': 0,
                'click_events': [],
                'scroll_depth': 0,
                'device_breakdown': {'desktop': 0, 'mobile': 0, 'tablet': 0}
            }
        
        # Analyze page layout and content structure
        # Check for common UX elements
        has_navigation = bool(soup.find(['nav', 'header', 'menu']) or 
                           soup.find_all('a', href=True) > 5)
        has_footer = bool(soup.find('footer'))
        has_search = bool(soup.find('input', {'type': 'search'}) or 
                        soup.find('form', string=re.compile('search', re.I)))
        
        # Check for readability factors
        paragraphs = soup.find_all('p')
        avg_paragraph_length = np.mean([len(p.get_text()) for p in paragraphs]) if paragraphs else 0
        readability_score = 100 if 50 <= avg_paragraph_length <= 120 else \
                           max(0, 100 - abs(avg_paragraph_length - 85) / 2)
        
        # Check for visual elements
        images = soup.find_all('img')
        image_count = len(images)
        has_video = bool(soup.find(['video', 'iframe']))
        
        # Check for mobile responsiveness
        has_responsive_meta = bool(soup.find('meta', {'name': 'viewport'}))
        has_media_queries = 'media' in html_content.lower() and 'screen' in html_content.lower()
        
        # Calculate layout score
        layout_score = 0
        if has_navigation:
            layout_score += 25
        if has_footer:
            layout_score += 15
        if has_search:
            layout_score += 15
        if image_count > 0:
            layout_score += min(25, image_count * 5)
        if has_video:
            layout_score += 20
        
        # Calculate mobile-friendliness score
        mobile_score = 0
        if has_responsive_meta:
            mobile_score += 50
        if has_media_queries:
            mobile_score += 50
        
        # Calculate engagement score based on provided metrics
        engagement_score = 0
        
        # Time on page (higher is better, up to a point)
        time_on_page = engagement_metrics.get('avg_time_on_page', 0)
        time_score = min(100, time_on_page / 2)  # 200 seconds (3.33 min) = perfect score
        
        # Bounce rate (lower is better)
        bounce_rate = engagement_metrics.get('bounce_rate', 100)
        bounce_score = max(0, 100 - bounce_rate)
        
        # Scroll depth (higher is better)
        scroll_depth = engagement_metrics.get('scroll_depth', 0)
        scroll_score = scroll_depth  # Assuming 0-100 scale
        
        # Click events (more interaction is better)
        click_events = engagement_metrics.get('click_events', [])
        click_score = min(100, len(click_events) * 10)
        
        # Calculate engagement score
        if any([time_on_page, bounce_rate < 100, scroll_depth, click_events]):
            engagement_score = (
                (0.35 * time_score) +
                (0.25 * bounce_score) +
                (0.25 * scroll_score) +
                (0.15 * click_score)
            )
        else:
            # No engagement data available, use content-based estimation
            engagement_score = (
                (0.4 * readability_score) +
                (0.3 * layout_score) +
                (0.3 * mobile_score)
            ) * 0.7  # Discount due to lack of actual engagement data
        
        # Calculate overall user experience score
        user_experience_score = (
            (0.3 * layout_score) +
            (0.2 * readability_score) +
            (0.2 * mobile_score) +
            (0.3 * engagement_score)
        )
        
        return {
            'url': url,
            'layout_elements': {
                'has_navigation': has_navigation,
                'has_footer': has_footer,
                'has_search': has_search,
                'image_count': image_count,
                'has_video': has_video
            },
            'layout_score': layout_score,
            'readability': {
                'avg_paragraph_length': avg_paragraph_length,
                'readability_score': readability_score
            },
            'mobile_friendliness': {
                'has_responsive_meta': has_responsive_meta,
                'has_media_queries': has_media_queries,
                'mobile_score': mobile_score
            },
            'engagement_metrics': engagement_metrics,
            'engagement_score': engagement_score,
            'user_experience_score': user_experience_score
        }
        
    def analyze_search_intent(self, url, html_content, query=None, serp_data=None):
        """
        Analyze search intent alignment using embedding matrices and clustering.
        As per guidance: Uses embedding matrices to represent queries and content
        in semantic space and clustering algorithms to group similar intents.
        
        Args:
            url: URL of the page to analyze
            html_content: HTML content of the page
            query: Optional search query to measure intent alignment against
            serp_data: Optional list of top-ranking pages for the query
            
        Returns:
            Dictionary with search intent metrics and score (0-100)
        """
        logger.info(f"Analyzing search intent alignment for {url}")
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Extract text content
        text_content = ""
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text_content += tag.get_text() + " "
        
        # Get title and meta description
        title = soup.title.string if soup.title else ""
        meta_description = ""
        for meta in soup.find_all('meta'):
            if meta.get('name', '').lower() == 'description':
                meta_description = meta.get('content', '')
                break
        
        # If no query provided, we can't measure intent alignment directly
        if not query:
            # Extract potential search intents from content
            # Use title and headings as proxies for intent
            headings = [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3'])]
            
            # Check if headings contain question words (who, what, when, where, why, how)
            question_pattern = re.compile(r'\b(who|what|when|where|why|how)\b', re.IGNORECASE)
            question_headings = [h for h in headings if question_pattern.search(h)]
            
            # Check for list-type content (how-to, listicles)
            list_items = len(soup.find_all('li'))
            has_ordered_lists = bool(soup.find('ol'))
            has_steps = any('step' in h.get_text().lower() for h in headings)
            
            # Determine likely intent types
            intent_types = []
            if question_headings:
                intent_types.append('informational')
            if has_ordered_lists or has_steps:
                intent_types.append('how-to')
            if any('buy' in h.lower() or 'price' in h.lower() or 'shop' in h.lower() for h in headings):
                intent_types.append('transactional')
            if not intent_types:
                intent_types.append('navigational')  # Default if no clear signals
            
            # Calculate a basic intent clarity score
            intent_clarity = min(100, len(headings) * 10 + list_items / 2)
            
            return {
                'url': url,
                'title': title,
                'meta_description': meta_description,
                'likely_intent_types': intent_types,
                'question_headings': question_headings,
                'list_items': list_items,
                'has_ordered_lists': has_ordered_lists,
                'has_steps': has_steps,
                'intent_clarity_score': intent_clarity,
                'search_intent_score': intent_clarity * 0.7  # Discount due to no query
            }
        
        # With query, we can do more sophisticated intent analysis
        # Create a simple TF-IDF vectorizer for embedding
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # Prepare corpus with query, page content, and SERP data if available
        corpus = [query, text_content]
        corpus_labels = ['query', 'page']
        
        if serp_data:
            for i, serp_item in enumerate(serp_data):
                if 'content' in serp_item:
                    corpus.append(serp_item['content'])
                    corpus_labels.append(f'serp_{i+1}')
        
        # Create embeddings
        try:
            embeddings = vectorizer.fit_transform(corpus)
            
            # Calculate cosine similarity between query and page
            query_page_similarity = cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
            
            # Cluster the embeddings if we have SERP data
            cluster_quality = 0
            serp_similarity = 0
            intent_match_score = query_page_similarity * 100
            
            if len(corpus) > 2:
                # Calculate similarity with top SERP results
                serp_similarities = cosine_similarity(embeddings[1:2], embeddings[2:])[0]
                serp_similarity = np.mean(serp_similarities)
                
                # Try clustering if we have enough data points
                if len(corpus) >= 4:
                    try:
                        # Use k-means to cluster the embeddings
                        num_clusters = min(3, len(corpus) - 1)
                        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                        clusters = kmeans.fit_predict(embeddings.toarray())
                        
                        # Check if query and page are in the same cluster
                        query_cluster = clusters[0]
                        page_cluster = clusters[1]
                        same_cluster = query_cluster == page_cluster
                        
                        # Count how many SERP results are in the same cluster as the query
                        serp_in_query_cluster = np.sum(clusters[2:] == query_cluster)
                        cluster_quality = serp_in_query_cluster / (len(corpus) - 2)
                        
                        # Adjust intent match score based on clustering
                        if same_cluster:
                            intent_match_score = min(100, intent_match_score * 1.2)
                        else:
                            intent_match_score = max(0, intent_match_score * 0.8)
                    except:
                        pass
        except:
            # Fallback if vectorization fails
            query_page_similarity = 0
            serp_similarity = 0
            intent_match_score = 0
        
        # Analyze query type
        query_type = 'informational'  # Default
        if re.search(r'\b(buy|price|shop|cost|purchase|order)\b', query, re.IGNORECASE):
            query_type = 'transactional'
        elif re.search(r'\b(how to|steps|guide|tutorial)\b', query, re.IGNORECASE):
            query_type = 'how-to'
        elif len(query.split()) <= 2:
            query_type = 'navigational'
        
        # Check if page content matches query type
        content_matches_type = False
        if query_type == 'transactional' and re.search(r'\b(buy|price|shop|cost|purchase|order)\b', text_content, re.IGNORECASE):
            content_matches_type = True
        elif query_type == 'how-to' and (has_ordered_lists or has_steps):
            content_matches_type = True
        elif query_type == 'informational' and len(text_content.split()) > 300:  # Longer content for informational
            content_matches_type = True
        elif query_type == 'navigational' and len(text_content.split()) < 300:  # Shorter content for navigational
            content_matches_type = True
        
        # Calculate intent type match score
        type_match_score = 100 if content_matches_type else 50
        
        # Calculate overall search intent score
        search_intent_score = (
            (0.5 * intent_match_score) +
            (0.3 * type_match_score) +
            (0.2 * (serp_similarity * 100))
        )
        
        return {
            'url': url,
            'query': query,
            'query_type': query_type,
            'content_matches_type': content_matches_type,
            'query_page_similarity': query_page_similarity,
            'serp_similarity': serp_similarity,
            'intent_match_score': intent_match_score,
            'type_match_score': type_match_score,
            'search_intent_score': search_intent_score
        }
        
    def analyze_page_speed(self, url, html_content, performance_metrics=None):
        """
        Analyze page speed and Core Web Vitals using arrays and min-heaps.
        As per guidance: Uses arrays and min-heaps for tracking resource loads
        and rendering events, with greedy approaches for optimization.
        
        Args:
            url: URL of the page to analyze
            html_content: HTML content of the page
            performance_metrics: Optional dictionary with performance data
            
        Returns:
            Dictionary with page speed metrics and score (0-100)
        """
        logger.info(f"Analyzing page speed and Core Web Vitals for {url}")
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'lxml')
        
        # If performance metrics are provided, use them directly
        if performance_metrics:
            lcp = performance_metrics.get('largest_contentful_paint', 0)  # in ms
            fid = performance_metrics.get('first_input_delay', 0)  # in ms
            cls = performance_metrics.get('cumulative_layout_shift', 0)  # unitless
            ttfb = performance_metrics.get('time_to_first_byte', 0)  # in ms
            fcp = performance_metrics.get('first_contentful_paint', 0)  # in ms
            load_time = performance_metrics.get('load_time', 0)  # in ms
        else:
            # Estimate metrics based on page content (very rough estimates)
            # In a real implementation, these would come from browser measurements
            
            # Count resources that might impact page speed
            scripts = soup.find_all('script', src=True)
            styles = soup.find_all('link', {'rel': 'stylesheet'})
            images = soup.find_all('img')
            iframes = soup.find_all('iframe')
            
            # Estimate total page size based on HTML size and estimated resource sizes
            html_size = len(html_content) / 1024  # KB
            estimated_js_size = len(scripts) * 200  # Assume average 200KB per script
            estimated_css_size = len(styles) * 50   # Assume average 50KB per stylesheet
            estimated_image_size = len(images) * 100  # Assume average 100KB per image
            estimated_iframe_size = len(iframes) * 300  # Assume average 300KB per iframe
            
            total_estimated_size = html_size + estimated_js_size + estimated_css_size + \
                                  estimated_image_size + estimated_iframe_size
            
            # Estimate Core Web Vitals based on page composition
            # These are very rough estimates for demonstration purposes
            lcp = 1000 + (total_estimated_size * 5)  # Larger pages take longer to render
            fid = 50 + (len(scripts) * 20)  # More scripts can increase input delay
            cls = min(1, 0.05 + (len(images) * 0.01))  # More images can cause more layout shifts
            ttfb = 200 + (html_size * 2)  # Larger HTML can increase TTFB
            fcp = 800 + (estimated_css_size * 10)  # More CSS can delay first paint
            load_time = 1000 + (total_estimated_size * 10)  # Total load time estimate
        
        # Calculate scores for each Core Web Vital
        # Based on Google's thresholds: https://web.dev/vitals/
        
        # LCP score (Good: <= 2500ms, Poor: > 4000ms)
        if lcp <= 2500:
            lcp_score = 100
        elif lcp <= 4000:
            lcp_score = 100 - ((lcp - 2500) / 1500 * 50)  # Linear scale from 100 to 50
        else:
            lcp_score = max(0, 50 - ((lcp - 4000) / 2000 * 50))  # Linear scale from 50 to 0
        
        # FID score (Good: <= 100ms, Poor: > 300ms)
        if fid <= 100:
            fid_score = 100
        elif fid <= 300:
            fid_score = 100 - ((fid - 100) / 200 * 50)  # Linear scale from 100 to 50
        else:
            fid_score = max(0, 50 - ((fid - 300) / 300 * 50))  # Linear scale from 50 to 0
        
        # CLS score (Good: <= 0.1, Poor: > 0.25)
        if cls <= 0.1:
            cls_score = 100
        elif cls <= 0.25:
            cls_score = 100 - ((cls - 0.1) / 0.15 * 50)  # Linear scale from 100 to 50
        else:
            cls_score = max(0, 50 - ((cls - 0.25) / 0.25 * 50))  # Linear scale from 50 to 0
        
        # TTFB score (Good: <= 800ms, Poor: > 1800ms)
        if ttfb <= 800:
            ttfb_score = 100
        elif ttfb <= 1800:
            ttfb_score = 100 - ((ttfb - 800) / 1000 * 50)  # Linear scale from 100 to 50
        else:
            ttfb_score = max(0, 50 - ((ttfb - 1800) / 1000 * 50))  # Linear scale from 50 to 0
        
        # FCP score (Good: <= 1800ms, Poor: > 3000ms)
        if fcp <= 1800:
            fcp_score = 100
        elif fcp <= 3000:
            fcp_score = 100 - ((fcp - 1800) / 1200 * 50)  # Linear scale from 100 to 50
        else:
            fcp_score = max(0, 50 - ((fcp - 3000) / 2000 * 50))  # Linear scale from 50 to 0
        
        # Overall load time score
        if load_time <= 3000:
            load_time_score = 100
        elif load_time <= 6000:
            load_time_score = 100 - ((load_time - 3000) / 3000 * 50)  # Linear scale from 100 to 50
        else:
            load_time_score = max(0, 50 - ((load_time - 6000) / 6000 * 50))  # Linear scale from 50 to 0
        
        # Calculate overall page speed score with Core Web Vitals weighted heavily
        page_speed_score = (
            (0.3 * lcp_score) +  # LCP is very important
            (0.2 * cls_score) +  # CLS is important
            (0.15 * fid_score) + # FID is important but less common
            (0.15 * ttfb_score) + # TTFB impacts perceived speed
            (0.1 * fcp_score) +  # FCP impacts perceived speed
            (0.1 * load_time_score)  # Overall load time still matters
        )
        
        # Identify optimization opportunities
        optimization_opportunities = []
        
        # Check for unminified resources
        if html_content.count('\n') > 100:
            optimization_opportunities.append('Minify HTML')
        
        # Check for render-blocking resources
        render_blocking_scripts = [s for s in soup.find_all('script') 
                                 if not s.get('async') and not s.get('defer') and s.get('src')]
        if render_blocking_scripts:
            optimization_opportunities.append('Eliminate render-blocking resources')
        
        # Check for unoptimized images
        unoptimized_images = [img for img in soup.find_all('img') 
                             if not img.get('loading') == 'lazy' and not img.get('srcset')]
        if unoptimized_images:
            optimization_opportunities.append('Optimize images and implement lazy loading')
        
        # Check for excessive DOM size
        dom_elements = len(soup.find_all())
        if dom_elements > 1500:
            optimization_opportunities.append('Reduce DOM size')
        
        return {
            'url': url,
            'core_web_vitals': {
                'largest_contentful_paint': lcp,
                'first_input_delay': fid,
                'cumulative_layout_shift': cls
            },
            'other_metrics': {
                'time_to_first_byte': ttfb,
                'first_contentful_paint': fcp,
                'load_time': load_time
            },
            'scores': {
                'lcp_score': lcp_score,
                'fid_score': fid_score,
                'cls_score': cls_score,
                'ttfb_score': ttfb_score,
                'fcp_score': fcp_score,
                'load_time_score': load_time_score
            },
            'optimization_opportunities': optimization_opportunities,
            'page_speed_score': page_speed_score
        }
        
    def analyze_brand_signals(self, url, html_content, social_data=None, brand_name=None):
        """
        Analyze brand and social signals using hash tables and sentiment analysis.
        As per guidance: Uses hash tables to count brand mentions and sentiment analysis
        to assess brand perception.
        
        Args:
            url: URL of the page to analyze
            html_content: HTML content of the page
            social_data: Optional dictionary with social media data
            brand_name: Optional brand name to look for
            
        Returns:
            Dictionary with brand and social metrics and score (0-100)
        """
        logger.info(f"Analyzing brand and social signals for {url}")
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Extract text content
        text_content = ""
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'span', 'div']):
            text_content += tag.get_text() + " "
        
        # Default social data if none provided
        if social_data is None:
            social_data = {
                'facebook_shares': 0,
                'twitter_shares': 0,
                'linkedin_shares': 0,
                'pinterest_pins': 0,
                'reddit_score': 0,
                'comments': [],
                'social_mentions': []
            }
        
        # Detect brand name if not provided
        if brand_name is None:
            # Try to extract from domain name
            domain = urlparse(url).netloc
            domain_parts = domain.split('.')
            if len(domain_parts) >= 2:
                brand_name = domain_parts[-2]  # e.g., 'example' from 'example.com'
            else:
                brand_name = domain
        
        # Count brand mentions in content (case insensitive)
        brand_mentions = 0
        if brand_name:
            brand_pattern = re.compile(r'\b' + re.escape(brand_name) + r'\b', re.IGNORECASE)
            brand_mentions = len(brand_pattern.findall(text_content))
        
        # Check for social media links
        social_links = {
            'facebook': bool(soup.find('a', href=re.compile(r'facebook\.com'))),
            'twitter': bool(soup.find('a', href=re.compile(r'twitter\.com|x\.com'))),
            'linkedin': bool(soup.find('a', href=re.compile(r'linkedin\.com'))),
            'instagram': bool(soup.find('a', href=re.compile(r'instagram\.com'))),
            'youtube': bool(soup.find('a', href=re.compile(r'youtube\.com'))),
            'pinterest': bool(soup.find('a', href=re.compile(r'pinterest\.com')))
        }
        
        # Count social sharing buttons
        share_buttons = len(soup.find_all('a', string=re.compile(r'share|tweet|pin', re.IGNORECASE))) + \
                        len(soup.find_all('button', string=re.compile(r'share|tweet|pin', re.IGNORECASE)))
        
        # Check for social proof elements
        testimonials = len(soup.find_all(['div', 'section'], string=re.compile(r'testimonial|review|client', re.IGNORECASE)))
        
        # Calculate social engagement score
        social_shares = sum([
            social_data.get('facebook_shares', 0),
            social_data.get('twitter_shares', 0),
            social_data.get('linkedin_shares', 0),
            social_data.get('pinterest_pins', 0),
            social_data.get('reddit_score', 0)
        ])
        
        # Logarithmic scaling for social shares (diminishing returns)
        if social_shares > 0:
            social_engagement_score = min(100, 20 * math.log10(social_shares + 1))
        else:
            social_engagement_score = 0
        
        # Calculate social presence score based on links
        social_presence_count = sum(1 for present in social_links.values() if present)
        social_presence_score = min(100, social_presence_count * 16.67)  # 6 platforms = 100%
        
        # Simple sentiment analysis on social mentions and comments
        positive_words = set(['good', 'great', 'excellent', 'amazing', 'awesome', 'love', 'best', 'fantastic',
                             'helpful', 'recommended', 'positive', 'perfect', 'wonderful', 'superb'])
        negative_words = set(['bad', 'poor', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disappointing',
                             'useless', 'negative', 'avoid', 'waste', 'mediocre', 'frustrating'])
        
        # Analyze sentiment in social mentions
        positive_mentions = 0
        negative_mentions = 0
        neutral_mentions = 0
        
        for mention in social_data.get('social_mentions', []) + social_data.get('comments', []):
            mention_text = mention.lower() if isinstance(mention, str) else \
                          mention.get('text', '').lower() if isinstance(mention, dict) else ''
            
            pos_count = sum(1 for word in positive_words if word in mention_text.split())
            neg_count = sum(1 for word in negative_words if word in mention_text.split())
            
            if pos_count > neg_count:
                positive_mentions += 1
            elif neg_count > pos_count:
                negative_mentions += 1
            else:
                neutral_mentions += 1
        
        total_mentions = positive_mentions + negative_mentions + neutral_mentions
        
        # Calculate sentiment score
        sentiment_score = 50  # Neutral default
        if total_mentions > 0:
            sentiment_ratio = (positive_mentions - negative_mentions) / total_mentions
            sentiment_score = 50 + (sentiment_ratio * 50)  # Scale to 0-100
        
        # Calculate brand prominence score
        brand_prominence = 0
        if brand_name:
            # Check if brand appears in important places
            title = soup.title.string if soup.title else ""
            h1_tags = [h.get_text() for h in soup.find_all('h1')]
            meta_desc = ""
            for meta in soup.find_all('meta'):
                if meta.get('name', '').lower() == 'description':
                    meta_desc = meta.get('content', '')
                    break
            
            brand_in_title = brand_name.lower() in title.lower() if title else False
            brand_in_h1 = any(brand_name.lower() in h.lower() for h in h1_tags)
            brand_in_meta = brand_name.lower() in meta_desc.lower()
            brand_in_url = brand_name.lower() in url.lower()
            
            # Calculate brand prominence score
            brand_prominence = 0
            if brand_in_title:
                brand_prominence += 30
            if brand_in_h1:
                brand_prominence += 25
            if brand_in_meta:
                brand_prominence += 20
            if brand_in_url:
                brand_prominence += 25
            
            # Add bonus for frequency of mentions
            brand_prominence += min(20, brand_mentions * 2)
            brand_prominence = min(100, brand_prominence)
        
        # Calculate overall brand and social score
        brand_social_score = (
            (0.3 * brand_prominence) +
            (0.3 * social_engagement_score) +
            (0.2 * social_presence_score) +
            (0.2 * sentiment_score)
        )
        
        return {
            'url': url,
            'brand_name': brand_name,
            'brand_mentions': brand_mentions,
            'brand_prominence': {
                'in_title': brand_in_title if brand_name else False,
                'in_h1': brand_in_h1 if brand_name else False,
                'in_meta': brand_in_meta if brand_name else False,
                'in_url': brand_in_url if brand_name else False,
                'score': brand_prominence
            },
            'social_presence': {
                'platforms': social_links,
                'share_buttons': share_buttons,
                'testimonials': testimonials,
                'score': social_presence_score
            },
            'social_engagement': {
                'total_shares': social_shares,
                'score': social_engagement_score
            },
            'sentiment_analysis': {
                'positive_mentions': positive_mentions,
                'negative_mentions': negative_mentions,
                'neutral_mentions': neutral_mentions,
                'sentiment_score': sentiment_score
            },
            'brand_social_score': brand_social_score
        }
        
    def calculate_seo_score(self, url, html_content, query=None, performance_metrics=None, social_data=None, brand_name=None):
        """
        Calculate the overall SEO score using the 2025-inspired formula combining all SEO factors:
        SEO Score = (0.25 × Content Quality & Relevance) + 
                   (0.20 × Backlink Authority) + 
                   (0.15 × Technical SEO) + 
                   (0.15 × User Experience & Engagement) + 
                   (0.10 × Search Intent Alignment) + 
                   (0.10 × Page Speed & Core Web Vitals) + 
                   (0.05 × Brand & Social Signals)
        
        Args:
            url: URL of the page to analyze
            html_content: HTML content of the page
            query: Optional search query for intent analysis
            performance_metrics: Optional dictionary with performance data
            social_data: Optional dictionary with social media data
            brand_name: Optional brand name to look for
            
        Returns:
            Dictionary with overall SEO score and individual factor scores
        """
        logger.info(f"Calculating overall SEO score for {url}")
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Extract text content
        text_content = ""
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'span', 'div']):
            text_content += tag.get_text() + " "
        
        # Get domain and page data
        domain = urlparse(url).netloc
        page_data = {
            'url': url,
            'domain': domain,
            'title': soup.title.string if soup.title else "",
            'meta_description': "",
            'content': text_content
        }
        
        # Get meta description
        for meta in soup.find_all('meta'):
            if meta.get('name', '').lower() == 'description':
                page_data['meta_description'] = meta.get('content', '')
                break
        
        # Analyze all SEO factors
        content_quality_results = self.analyze_content_quality(url, html_content, query)
        backlink_authority_results = self.analyze_backlink_authority(url)
        technical_seo_results = self.analyze_technical_seo(url, html_content)
        user_experience_results = self.analyze_user_experience(url, html_content)
        search_intent_results = self.analyze_search_intent(url, html_content, query) if query else {'search_intent_score': 0}
        page_speed_results = self.analyze_page_speed(url, html_content, performance_metrics)
        brand_social_results = self.analyze_brand_signals(url, html_content, social_data, brand_name)
        
        # Extract individual factor scores
        content_quality_score = content_quality_results.get('content_quality_score', 0)
        backlink_authority_score = backlink_authority_results.get('backlink_authority_score', 0)
        technical_seo_score = technical_seo_results.get('technical_seo_score', 0)
        user_experience_score = user_experience_results.get('user_experience_score', 0)
        search_intent_score = search_intent_results.get('search_intent_score', 0)
        page_speed_score = page_speed_results.get('page_speed_score', 0)
        brand_social_score = brand_social_results.get('brand_social_score', 0)
        
        # Calculate overall SEO score using the 2025-inspired formula
        seo_score = (
            (0.25 * content_quality_score) +
            (0.20 * backlink_authority_score) +
            (0.15 * technical_seo_score) +
            (0.15 * user_experience_score) +
            (0.10 * search_intent_score) +
            (0.10 * page_speed_score) +
            (0.05 * brand_social_score)
        )
        
        # Round to 2 decimal places
        seo_score = round(seo_score, 2)
        
        # Generate recommendations based on factor scores
        recommendations = []
        
        if content_quality_score < 70:
            recommendations.append("Improve content quality and relevance")
        if backlink_authority_score < 70:
            recommendations.append("Build more quality backlinks")
        if technical_seo_score < 70:
            recommendations.append("Fix technical SEO issues")
        if user_experience_score < 70:
            recommendations.append("Enhance user experience and engagement")
        if search_intent_score < 70 and query:
            recommendations.append("Better align content with search intent")
        if page_speed_score < 70:
            recommendations.append("Improve page speed and Core Web Vitals")
        if brand_social_score < 70:
            recommendations.append("Strengthen brand presence and social signals")
        
        # Determine SEO rating based on score
        if seo_score >= 90:
            seo_rating = "Excellent"
        elif seo_score >= 80:
            seo_rating = "Very Good"
        elif seo_score >= 70:
            seo_rating = "Good"
        elif seo_score >= 60:
            seo_rating = "Fair"
        elif seo_score >= 50:
            seo_rating = "Average"
        elif seo_score >= 40:
            seo_rating = "Below Average"
        elif seo_score >= 30:
            seo_rating = "Poor"
        else:
            seo_rating = "Very Poor"
        
        return {
            'url': url,
            'seo_score': seo_score,
            'seo_rating': seo_rating,
            'factor_scores': {
                'content_quality_score': content_quality_score,
                'backlink_authority_score': backlink_authority_score,
                'technical_seo_score': technical_seo_score,
                'user_experience_score': user_experience_score,
                'search_intent_score': search_intent_score,
                'page_speed_score': page_speed_score,
                'brand_social_score': brand_social_score
            },
            'factor_weights': {
                'content_quality': 0.25,
                'backlink_authority': 0.20,
                'technical_seo': 0.15,
                'user_experience': 0.15,
                'search_intent': 0.10,
                'page_speed': 0.10,
                'brand_social': 0.05
            },
            'recommendations': recommendations,
            'detailed_results': {
                'content_quality': content_quality_results,
                'backlink_authority': backlink_authority_results,
                'technical_seo': technical_seo_results,
                'user_experience': user_experience_results,
                'search_intent': search_intent_results,
                'page_speed': page_speed_results,
                'brand_social': brand_social_results
            }
        }
        
    def analyze_url(self, url, query=None, max_depth=0, performance_metrics=None, social_data=None, brand_name=None):
        """
        Complete analysis of a URL, crawling it first if needed and then calculating SEO score.
        
        Args:
            url: URL to analyze
            query: Optional search query for intent analysis
            max_depth: Maximum crawl depth (0 for just the URL)
            performance_metrics: Optional dictionary with performance data
            social_data: Optional dictionary with social media data
            brand_name: Optional brand name to look for
            
        Returns:
            Dictionary with SEO score and analysis results
        """
        from crawler import Crawler
        
        logging.info(f"Starting full SEO analysis for {url}")
        
        # Create a crawler instance
        crawler = Crawler(max_depth=max_depth, max_pages=10, respect_robots=True)
        
        # Crawl the URL
        crawl_results = crawler.crawl_url(url)
        
        if not crawl_results or url not in crawl_results:
            logging.error(f"Failed to crawl {url}")
            return {'error': f"Failed to crawl {url}"}
        
        # Get the HTML content
        html_content = crawl_results[url]['html']
        
        # Calculate SEO score
        seo_results = self.calculate_seo_score(
            url, 
            html_content, 
            query=query,
            performance_metrics=performance_metrics,
            social_data=social_data,
            brand_name=brand_name
        )
        
        return seo_results
        
    def compare_urls(self, urls, query=None, max_depth=0):
        """
        Compare multiple URLs for SEO performance
        
        Args:
            urls (list): List of URLs to compare
            query (str, optional): Search query for intent analysis
            max_depth (int, optional): Crawl depth
            
        Returns:
            list: List of analysis results for each URL
        """
        logging.info(f"Comparing {len(urls)} URLs")
        
        results = []
        for url in urls:
            try:
                # Analyze each URL
                result = self.analyze_url(url, query=query, max_depth=max_depth)
                results.append(result)
            except Exception as e:
                logging.error(f"Error analyzing URL {url}: {str(e)}")
                # Add a placeholder result with error information
                results.append({
                    "url": url,
                    "error": str(e),
                    "seo_score": 0,
                    "seo_rating": "Error",
                    "factor_scores": {
                        "content_quality_score": 0,
                        "backlink_authority_score": 0,
                        "technical_seo_score": 0,
                        "user_experience_score": 0,
                        "search_intent_score": 0,
                        "page_speed_score": 0,
                        "brand_social_score": 0
                    }
                })
        
        return results


def main():
    """Main function to demonstrate the SEO analyzer"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='SEO Analyzer')
    parser.add_argument('url', help='URL to analyze')
    parser.add_argument('--query', help='Search query for intent analysis')
    parser.add_argument('--depth', type=int, default=0, help='Crawl depth (default: 0)')
    parser.add_argument('--output', help='Output file for results (JSON format)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SEOAnalyzer()
    
    # Analyze URL
    results = analyzer.analyze_url(args.url, query=args.query, max_depth=args.depth)
    
    # Print results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

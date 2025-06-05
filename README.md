# Google-Like Web Crawler & SEO Analyzer

This project implements a comprehensive web crawler and SEO analysis system that:
- Crawls websites like Google's crawler
- Indexes content for analysis
- Provides a UI for analyzing SEO factors
- Calculates an SEO score based on multiple factors
- Supports batch analysis of multiple URLs
- Allows side-by-side comparison of websites

## SEO Scoring Formula (2025-Inspired)
```
SEO Score = (0.25 × Content Quality & Relevance) + 
            (0.20 × Backlink Authority) + 
            (0.15 × Technical SEO) + 
            (0.15 × User Experience & Engagement) + 
            (0.10 × Search Intent Alignment) + 
            (0.10 × Page Speed & Core Web Vitals) + 
            (0.05 × Brand & Social Signals)
```

## Project Components
- `crawler.py`: Web crawling functionality
- `indexer.py`: Content indexing and storage
- `analyzer.py`: SEO factor analysis with 7 key SEO factors
- `app.py`: Flask web application with UI routes
- `templates/`: HTML templates for the web interface
  - `index.html`: Homepage with URL input form
  - `results.html`: Detailed SEO analysis results
  - `batch.html`: Batch analysis upload form
  - `batch_results.html`: Results for multiple URLs
  - `compare.html`: URL comparison input form
  - `compare_results.html`: Side-by-side comparison results

## Setup Instructions
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Install Chrome/Firefox for browser automation
3. Run the application:
   ```
   python app.py
   ```
4. Access the UI at http://localhost:5000

## Usage

### Single URL Analysis
1. Enter a URL in the web interface
2. Set crawl depth and other parameters
3. Start the crawl and analysis process
4. View detailed SEO analysis and scores with visualizations

### Batch Analysis
1. Navigate to the Batch Analysis page
2. Upload a CSV file with URLs (and optional search queries)
3. Process multiple URLs at once
4. View aggregated results and individual scores

### URL Comparison
1. Navigate to the Compare URLs page
2. Enter multiple URLs (up to 5) to compare
3. Add an optional search query for intent analysis
4. View side-by-side comparison with radar charts and detailed metrics

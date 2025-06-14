# Google-Like Web Crawler, SEO & UI/UX Analyzer

This project implements a comprehensive web crawler with SEO and UI/UX analysis capabilities that:
- Crawls websites like Google's crawler
- Indexes content for analysis
- Provides a UI for analyzing SEO factors and UI/UX metrics
- Calculates an SEO score based on multiple factors
- Performs detailed UI/UX analysis including visual design, interaction design, and accessibility
- Supports advanced UI testing using headless browser automation
- Supports batch analysis of multiple URLs
- Allows side-by-side comparison of websites

## System Design & Architecture

### High-Level Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │     │                 │
│  Web Crawler    │────▶│  Indexer        │────▶│  SEO Analyzer   │────▶│  UI/UX Analyzer │
│  (crawler.py)   │     │  (indexer.py)   │     │  (analyzer.py)  │     │(ui_ux_analyzer.py)
│                 │     │                 │     │                 │     │                 │
└────────┬────────┘     └─────────────────┘     └────────┬────────┘     └────────┬────────┘
         │                                               │                        │
         │                                               │                        │
         │                                               │                        │
         │                                               ▼                        │
┌────────▼────────┐                           ┌─────────────────┐                  │
│                 │                           │                 │                  │
│  Web Pages      │                           │  SEO Analysis   │                  │
│  (HTML Content) │                           │  Results        │                  │
│                 │                           │                 │                  │
└─────────────────┘                           └────────┬────────┘                  │
                                                       │                          │
                                                       │                          │
                                                       ▼                          ▼
                                             ┌─────────────────────────────────────┐
                                             │                                     │
                                             │  Enhanced SEO Analyzer             │
                                             │  (ui_ux_integration.py)            │
                                             │                                     │
                                             └────────────────┬────────────────────┘
                                                              │
                                                              │
                                                              ▼
                                             ┌─────────────────────────────────────┐
                                             │                                     │
                                             │  Chainlit Web App                   │
                                             │  (chainlit_app.py)                  │
                                             │                                     │
                                             └────────────────┬────────────────────┘
                                                              │
                                                              │
                                                              ▼
                                             ┌─────────────────────────────────────┐
                                             │                                     │
                                             │  Interactive Web Interface          │
                                             │                                     │
                                             └─────────────────────────────────────┘
```

### Component Interactions
1. **Web Crawler**: Fetches web pages asynchronously, respects robots.txt, and extracts page data
2. **Indexer**: Processes and stores crawled content in efficient data structures
3. **SEO Analyzer**: Evaluates 7 key SEO factors using specialized algorithms
4. **UI/UX Analyzer**: Evaluates visual design, interaction design, accessibility, and advanced UI metrics
5. **Enhanced SEO Analyzer**: Integrates UI/UX analysis with SEO analysis for comprehensive evaluation
6. **Chainlit Web App**: Provides an interactive chat-based interface for analysis commands
7. **Interactive Web Interface**: Visualizes SEO and UI/UX analysis with interactive charts and comparisons

## Tools & Frameworks Used

### Backend
- **Python 3**: Core programming language
- **Chainlit**: Chat-based web application framework for interactive data apps
- **BeautifulSoup4**: HTML parsing and content extraction
- **aiohttp**: Asynchronous HTTP requests for efficient crawling
- **NetworkX**: Graph algorithms for backlink analysis and PageRank
- **NLTK**: Natural language processing for content analysis
- **scikit-learn**: Machine learning for clustering and similarity analysis
- **NumPy/Pandas**: Data manipulation and analysis
- **Selenium**: Headless browser automation for advanced UI testing
- **axe-selenium-python**: Accessibility testing with Selenium

### UI/UX Analysis
- **Selenium WebDriver**: Automated browser testing for UI/UX metrics
- **wcag-contrast-ratio**: Color contrast analysis for accessibility
- **colour**: Color manipulation and analysis
- **textstat**: Text readability metrics
- **spacy**: Advanced NLP for content analysis
- **Pillow**: Image processing and analysis
- **OpenCV**: Computer vision for visual design analysis

### Frontend
- **Chainlit UI**: Interactive chat-based user interface
- **Plotly**: Interactive data visualization (radar charts, bar charts, etc.)
- **Markdown**: Rich text formatting for analysis results

### Development & Deployment
- **Jinja2**: Templating engine for HTML generation
- **tqdm**: Progress bars for long-running operations
- **python-dotenv**: Environment variable management

## Data Structures & Algorithms

### Web Crawler
- **Breadth-First Search (BFS)**: URL crawling algorithm using deque for efficient frontier management
- **Hash Sets**: Track visited URLs to avoid duplicates
- **Hash Maps**: Store domain-specific crawl delays and robots.txt parsers
- **URL Normalization**: Canonicalize URLs to avoid duplicate content

### SEO Analyzer
- **Inverted Index (Hash Tables)**: For efficient content analysis and keyword indexing
- **Directed Graph & PageRank**: Models link structure and calculates authority scores
- **TF-IDF Vectorization**: Measures content relevance and keyword importance
- **Cosine Similarity**: Calculates semantic relevance between content and queries
- **K-Means Clustering**: Groups similar content for search intent analysis
- **Min-Heaps**: Track resource loads for page speed optimization
- **Bloom Filters**: Efficient duplicate content detection
- **Sentiment Analysis**: Evaluates brand perception in content

### Performance Optimizations
- **Asynchronous Crawling**: Non-blocking I/O for efficient web page fetching
- **Caching**: Stores frequently accessed data to reduce computation
- **Lazy Loading**: Defers resource-intensive operations until needed
- **Parallelization**: Distributes workload for batch analysis

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
- `ui_ux_analyzer.py`: UI/UX analysis including visual design, interaction design, accessibility, and advanced UI testing
- `ui_ux_integration.py`: Integration of UI/UX analysis with SEO analysis
- `chainlit_app.py`: Chainlit web application with interactive chat interface
- `chainlit.md`: Chainlit welcome page and documentation
- `.chainlit/`: Chainlit configuration directory
  - `config.toml`: Chainlit configuration settings

## Setup Instructions
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Install Chrome/Firefox for browser automation (required for UI/UX analysis)
3. Run the application:
   ```
   chainlit run chainlit_app.py
   ```
4. Access the UI at http://localhost:8000

## Usage

### Single URL Analysis
1. In the chat interface, type `/analyze [url] [query] [depth]`
   - Example: `/analyze https://example.com best seo practices 0`
2. View detailed SEO analysis and scores with interactive visualizations

### Enhanced SEO Analysis with UI/UX Metrics
1. In the chat interface, type `/enhanced [url] [query] [depth]`
   - Example: `/enhanced https://example.com seo optimization 1`
2. Add `advanced` at the end to run advanced UI tests (requires browser automation)
   - Example: `/enhanced https://example.com seo optimization 0 advanced`
3. View comprehensive analysis with both SEO and UI/UX metrics

### Dedicated UI/UX Analysis
1. In the chat interface, type `/ui-ux [url]`
   - Example: `/ui-ux https://example.com`
2. Add `advanced` to run advanced UI tests with Selenium
   - Example: `/ui-ux https://example.com advanced`
3. View detailed UI/UX analysis including visual design, interaction design, and accessibility metrics

### URL Comparison
1. In the chat interface, type `/compare [url1] [url2] ...`
   - Example: `/compare https://example.com https://another-site.com`
2. View side-by-side comparison with radar charts and detailed metrics

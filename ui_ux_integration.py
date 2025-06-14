#!/usr/bin/env python3
"""
UI/UX Integration module for web crawler project.
Connects the enhanced UI/UX analyzer with the existing SEO analyzer.
"""

import logging
from ui_ux_analyzer import UIUXAnalyzer
from analyzer import SEOAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedSEOAnalyzer(SEOAnalyzer):
    """
    Enhanced SEO Analyzer that incorporates advanced UI/UX analysis.
    Extends the base SEOAnalyzer class with improved user experience and brand signal analysis.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the enhanced SEO analyzer."""
        super().__init__(*args, **kwargs)
        self.ui_ux_analyzer = UIUXAnalyzer()
        logger.info("Enhanced SEO Analyzer initialized with UI/UX capabilities")
        
    def fetch_url(self, url):
        """
        Fetch HTML content from a URL.
        
        Args:
            url: URL to fetch content from
            
        Returns:
            HTML content of the page
        """
        import requests
        from requests.exceptions import RequestException
        
        # Try different user agents if the first one fails
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
            'SEOAnalyzerBot/1.0'
        ]
        
        # Common headers to mimic a browser
        common_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        last_error = None
        
        # Try each user agent
        for user_agent in user_agents:
            try:
                headers = {'User-Agent': user_agent}
                headers.update(common_headers)
                
                logger.info(f"Attempting to fetch {url} with user agent: {user_agent}")
                response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
                
                # Check if we got a successful response
                if response.status_code == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' in content_type:
                        logger.info(f"Successfully fetched HTML content from {url}")
                        return response.text
                    else:
                        logger.warning(f"URL {url} returned non-HTML content: {content_type}")
                        last_error = f"Non-HTML content: {content_type}"
                else:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
                    last_error = f"HTTP error: {response.status_code}"
                    
            except RequestException as e:
                logger.warning(f"Request error with user agent {user_agent} for {url}: {str(e)}")
                last_error = str(e)
            except Exception as e:
                logger.warning(f"Unexpected error with user agent {user_agent} for {url}: {str(e)}")
                last_error = str(e)
        
        # If we've tried all user agents and still failed
        error_msg = f"Failed to fetch URL after trying multiple user agents: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def analyze_ui_ux(self, url, run_advanced_tests=False):
        """
        Perform UI/UX analysis on the given URL.
        
        Args:
            url: URL of the page to analyze
            run_advanced_tests: Whether to run advanced browser-based tests
            
        Returns:
            Dictionary with comprehensive UI/UX analysis results
        """
        logger.info(f"Analyzing UI/UX for {url} with advanced tests: {run_advanced_tests}")
        
        try:
            # Fetch HTML content
            html_content = self.fetch_url(url)
            
            # Check if HTML content is None or empty
            if not html_content:
                raise Exception(f"Failed to retrieve HTML content from {url}. The site may be blocking requests or returning non-HTML content.")
            
            # Use the UI/UX analyzer to analyze the page
            ui_ux_results = self.ui_ux_analyzer.analyze_ui_ux(url, html_content, run_advanced_tests)
            
            return ui_ux_results
            
        except Exception as e:
            logger.error(f"Error in UI/UX analysis for {url}: {str(e)}")
            raise Exception(f"UI/UX analysis failed: {str(e)}")
        
    def analyze_user_experience_enhanced(self, url, html_content, engagement_metrics=None, run_advanced_tests=False):
        """
        Enhanced version of analyze_user_experience that incorporates the advanced UI/UX analysis.
        
        Args:
            url: URL of the page to analyze
            html_content: HTML content of the page
            engagement_metrics: Dictionary with engagement metrics (optional)
            run_advanced_tests: Whether to run advanced browser-based tests
            
        Returns:
            Dictionary with comprehensive user experience analysis and score (0-100)
        """
        logger.info(f"Analyzing enhanced user experience for {url}")
        
        # Get base user experience analysis from parent class
        base_ux_analysis = self.analyze_user_experience(url, html_content, engagement_metrics)
        
        # Get enhanced UI/UX analysis
        ui_ux_analysis = self.ui_ux_analyzer.analyze_ui_ux(url, html_content, run_advanced_tests)
        
        # Combine scores with appropriate weights
        # Base UX score gets 40% weight, enhanced UI/UX gets 60% weight
        combined_score = (base_ux_analysis['user_experience_score'] * 0.4) + (ui_ux_analysis['ui_ux_score'] * 0.6)
        
        # Merge recommendations
        all_recommendations = base_ux_analysis.get('recommendations', []) + ui_ux_analysis.get('recommendations', [])
        
        # Create combined analysis result
        enhanced_analysis = {
            'url': url,
            'base_user_experience': base_ux_analysis,
            'enhanced_ui_ux': ui_ux_analysis,
            'combined_user_experience_score': combined_score,
            'recommendations': all_recommendations
        }
        
        return enhanced_analysis
    
    def analyze_brand_signals_enhanced(self, url, html_content, social_data=None, brand_name=None):
        """
        Enhanced version of analyze_brand_signals that incorporates visual design analysis
        for brand consistency and recognition.
        
        Args:
            url: URL of the page to analyze
            html_content: HTML content of the page
            social_data: Dictionary with social engagement data (optional)
            brand_name: Name of the brand to analyze (optional)
            
        Returns:
            Dictionary with comprehensive brand signals analysis and score (0-100)
        """
        logger.info(f"Analyzing enhanced brand signals for {url}")
        
        # Get base brand signals analysis from parent class
        base_brand_analysis = self.analyze_brand_signals(url, html_content, social_data, brand_name)
        
        # Get visual design analysis from UI/UX analyzer
        visual_design = self.ui_ux_analyzer.analyze_visual_design(html_content)
        
        # Brand consistency is heavily influenced by visual design consistency
        brand_consistency_score = visual_design['consistency']['score']
        
        # Brand recognition is influenced by visual hierarchy and color harmony
        brand_recognition_score = (
            (visual_design['visual_hierarchy']['score'] * 0.5) +
            (visual_design['color_harmony']['score'] * 0.5)
        )
        
        # Combine scores with appropriate weights
        # Base brand score gets 60% weight, design-based brand factors get 40% weight
        combined_score = (
            (base_brand_analysis['brand_signals_score'] * 0.6) +
            (brand_consistency_score * 0.2) +
            (brand_recognition_score * 0.2)
        )
        
        # Create recommendations based on visual design scores
        recommendations = base_brand_analysis.get('recommendations', [])
        
        if visual_design['consistency']['score'] < 70:
            recommendations.append("Improve brand consistency through more uniform design elements")
            
        if visual_design['visual_hierarchy']['score'] < 70:
            recommendations.append("Enhance brand visibility by improving visual hierarchy")
            
        if visual_design['color_harmony']['score'] < 70:
            recommendations.append("Refine brand color scheme for better recognition and harmony")
        
        # Create combined analysis result
        enhanced_analysis = {
            'url': url,
            'base_brand_signals': base_brand_analysis,
            'brand_design_factors': {
                'brand_consistency': {
                    'score': brand_consistency_score,
                    'design_consistency': visual_design['consistency']
                },
                'brand_recognition': {
                    'score': brand_recognition_score,
                    'visual_hierarchy': visual_design['visual_hierarchy'],
                    'color_harmony': visual_design['color_harmony']
                }
            },
            'combined_brand_signals_score': combined_score,
            'recommendations': recommendations
        }
        
        return enhanced_analysis
    
    def perform_enhanced_analysis(self, url, html_content=None, run_advanced_tests=False, **kwargs):
        """
        Perform enhanced SEO analysis including advanced UI/UX metrics.
        
        Args:
            url: URL of the page to analyze
            html_content: HTML content of the page (optional, will be fetched if not provided)
            run_advanced_tests: Whether to run advanced browser-based tests
            **kwargs: Additional arguments to pass to the base analyze method
            
        Returns:
            Dictionary with comprehensive SEO analysis including enhanced UI/UX metrics
        """
        logger.info(f"Performing enhanced SEO analysis for {url}")
        
        # Get base SEO analysis from parent class
        base_analysis = self.analyze(url, html_content, **kwargs)
        
        # Replace user experience and brand signals with enhanced versions
        if html_content is None and 'html_content' in base_analysis:
            html_content = base_analysis['html_content']
        
        # Get enhanced analyses
        enhanced_ux = self.analyze_user_experience_enhanced(
            url, 
            html_content, 
            kwargs.get('engagement_metrics'),
            run_advanced_tests
        )
        
        enhanced_brand = self.analyze_brand_signals_enhanced(
            url, 
            html_content, 
            kwargs.get('social_data'),
            kwargs.get('brand_name')
        )
        
        # Update the base analysis with enhanced components
        enhanced_analysis = base_analysis.copy()
        enhanced_analysis['user_experience'] = enhanced_ux
        enhanced_analysis['brand_signals'] = enhanced_brand
        
        # Recalculate overall SEO score with enhanced components
        if 'seo_score' in base_analysis:
            # Assume user experience and brand signals each contribute 15% to overall score
            # Remove their original contribution and add enhanced contribution
            other_factors_weight = 0.7  # 100% - 15% - 15%
            other_factors_score = (base_analysis['seo_score'] - 
                                  (base_analysis['user_experience']['user_experience_score'] * 0.15) -
                                  (base_analysis['brand_signals']['brand_signals_score'] * 0.15)) / other_factors_weight
            
            enhanced_analysis['seo_score'] = (
                (other_factors_score * other_factors_weight) +
                (enhanced_ux['combined_user_experience_score'] * 0.15) +
                (enhanced_brand['combined_brand_signals_score'] * 0.15)
            )
        
        # Add UI/UX specific recommendations to overall recommendations
        if 'recommendations' in base_analysis:
            enhanced_analysis['recommendations'] = base_analysis['recommendations'] + [
                rec for rec in enhanced_ux.get('recommendations', []) + enhanced_brand.get('recommendations', [])
                if rec not in base_analysis['recommendations']
            ]
        
        return enhanced_analysis

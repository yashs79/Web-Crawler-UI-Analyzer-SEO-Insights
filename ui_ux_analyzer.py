#!/usr/bin/env python3
"""
Enhanced UI/UX Analyzer module for web crawler project.
Implements advanced UI/UX analysis techniques including visual design analysis,
interaction design evaluation, accessibility testing, and more.
"""

import os
import re
import logging
import json
from urllib.parse import urlparse, urljoin
import numpy as np
from bs4 import BeautifulSoup
import textstat
import wcag_contrast_ratio
from colour import Color

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UIUXAnalyzer:
    """
    Enhanced UI/UX Analyzer that evaluates various aspects of user interface
    and user experience of web pages.
    """
    
    def __init__(self):
        """Initialize the UI/UX analyzer."""
        self.accessibility_weights = {
            'color_contrast': 0.25,
            'alt_text': 0.20,
            'aria_attributes': 0.15,
            'keyboard_nav': 0.20,
            'form_labels': 0.20
        }
        
        self.visual_design_weights = {
            'color_harmony': 0.20,
            'visual_hierarchy': 0.25,
            'consistency': 0.25,
            'whitespace': 0.15,
            'typography': 0.15
        }
        
        self.interaction_design_weights = {
            'form_usability': 0.30,
            'cta_effectiveness': 0.25,
            'navigation_paths': 0.25,
            'feedback_mechanisms': 0.20
        }
        
    def analyze_interaction_design(self, html_content):
        """
        Analyze interaction design aspects of the webpage.
        
        Args:
            html_content: HTML content of the page
            
        Returns:
            Dictionary with interaction design metrics and score (0-100)
        """
        logger.info("Analyzing interaction design")
        
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Form usability analysis
        forms = soup.find_all('form')
        form_usability_score = 0
        form_analysis = {
            'total_forms': len(forms),
            'forms_with_labels': 0,
            'forms_with_validation': 0,
            'forms_with_submit': 0,
            'forms_with_placeholders': 0
        }
        
        if forms:
            for form in forms:
                # Check for proper labels
                inputs = form.find_all(['input', 'select', 'textarea'])
                labeled_inputs = 0
                
                for input_field in inputs:
                    # Check if input has an id and there's a label with for=id
                    if input_field.get('id'):
                        label = soup.find('label', {'for': input_field['id']})
                        if label:
                            labeled_inputs += 1
                    # Check for placeholder as alternative
                    if input_field.get('placeholder'):
                        form_analysis['forms_with_placeholders'] += 1
                
                if inputs and labeled_inputs / len(inputs) >= 0.7:  # 70% of inputs have labels
                    form_analysis['forms_with_labels'] += 1
                
                # Check for client-side validation
                if form.find('input', {'required': True}) or 'validate' in str(form) or 'validation' in str(form):
                    form_analysis['forms_with_validation'] += 1
                
                # Check for submit button
                if form.find('button', {'type': 'submit'}) or form.find('input', {'type': 'submit'}):
                    form_analysis['forms_with_submit'] += 1
            
            # Calculate form usability score
            if form_analysis['total_forms'] > 0:
                labels_ratio = form_analysis['forms_with_labels'] / form_analysis['total_forms']
                validation_ratio = form_analysis['forms_with_validation'] / form_analysis['total_forms']
                submit_ratio = form_analysis['forms_with_submit'] / form_analysis['total_forms']
                
                form_usability_score = (
                    (labels_ratio * 40) +
                    (validation_ratio * 30) +
                    (submit_ratio * 30)
                )
        else:
            # No forms to analyze
            form_usability_score = 50  # Neutral score
        
        # Call-to-action (CTA) effectiveness
        cta_score = 0
        cta_elements = []
        
        # Look for buttons with action words
        action_words = ['buy', 'download', 'get', 'sign up', 'register', 'subscribe', 'try', 'start', 'join', 'learn more']
        for word in action_words:
            cta_elements.extend(soup.find_all(['a', 'button'], string=re.compile(word, re.I)))
        
        # Look for elements with CTA-like classes
        cta_elements.extend(soup.find_all(['a', 'button'], class_=re.compile(r'cta|call-to-action|primary', re.I)))
        
        # Deduplicate
        cta_elements = list(set(cta_elements))
        
        # Analyze CTAs
        if cta_elements:
            # Check for visibility (size, color, position)
            visible_ctas = 0
            for cta in cta_elements:
                # Check for style attributes that suggest visibility
                if cta.get('style') and ('color' in cta['style'] or 'background' in cta['style'] or 
                                       'font-size' in cta['style'] or 'padding' in cta['style']):
                    visible_ctas += 1
                # Check for classes that suggest visibility
                elif cta.get('class') and any('btn' in cls.lower() or 'button' in cls.lower() for cls in cta['class']):
                    visible_ctas += 1
            
            # Calculate CTA score
            cta_ratio = visible_ctas / len(cta_elements) if cta_elements else 0
            cta_count_score = min(100, len(cta_elements) * 20)  # Up to 5 CTAs get full score
            
            cta_score = (cta_ratio * 50) + (cta_count_score * 0.5)
        else:
            # No clear CTAs found
            cta_score = 30  # Penalty for no clear CTAs
        
        # Navigation paths analysis
        nav_score = 0
        navigation = soup.find('nav')
        menu = navigation or soup.find(class_=re.compile(r'menu|navigation', re.I))
        
        if menu:
            # Check for hierarchical navigation
            nav_items = menu.find_all('li')
            nav_links = menu.find_all('a')
            
            # Check for dropdown menus (nested lists)
            has_dropdowns = bool(menu.find('ul', recursive=True))
            
            # Calculate navigation score
            if nav_links:
                nav_score = 50  # Base score for having navigation
                
                # Bonus for having a good number of nav items (not too many, not too few)
                if 3 <= len(nav_links) <= 10:
                    nav_score += 20
                
                # Bonus for hierarchical navigation
                if has_dropdowns:
                    nav_score += 20
                
                # Check if navigation has current page indicator
                if soup.find('a', class_=re.compile(r'active|current', re.I)):
                    nav_score += 10
        else:
            # No clear navigation
            nav_score = 20  # Penalty for no clear navigation
        
        # Feedback mechanisms analysis
        feedback_score = 0
        
        # Check for form feedback
        has_form_validation = bool(soup.find('input', {'required': True})) or 'validate' in html_content
        
        # Check for success/error messages
        has_alerts = bool(soup.find(class_=re.compile(r'alert|message|notification|toast', re.I)))
        
        # Check for loading indicators
        has_loading = bool(soup.find(class_=re.compile(r'loading|spinner|progress', re.I)))
        
        # Calculate feedback score
        feedback_score = 0
        if has_form_validation:
            feedback_score += 40
        if has_alerts:
            feedback_score += 30
        if has_loading:
            feedback_score += 30
        
        # Calculate overall interaction design score
        interaction_design_score = (
            (self.interaction_design_weights['form_usability'] * form_usability_score) +
            (self.interaction_design_weights['cta_effectiveness'] * cta_score) +
            (self.interaction_design_weights['navigation_paths'] * nav_score) +
            (self.interaction_design_weights['feedback_mechanisms'] * feedback_score)
        )
        
        return {
            'form_usability': {
                'analysis': form_analysis,
                'score': form_usability_score
            },
            'cta_effectiveness': {
                'cta_count': len(cta_elements),
                'score': cta_score
            },
            'navigation_paths': {
                'has_navigation': bool(menu),
                'nav_links_count': len(menu.find_all('a')) if menu else 0,
                'has_dropdowns': has_dropdowns if menu else False,
                'score': nav_score
            },
            'feedback_mechanisms': {
                'has_form_validation': has_form_validation,
                'has_alerts': has_alerts,
                'has_loading': has_loading,
                'score': feedback_score
            },
            'interaction_design_score': interaction_design_score
        }
        
    def analyze_accessibility(self, html_content):
        """
        Analyze accessibility aspects of the webpage according to WCAG guidelines.
        
        Args:
            html_content: HTML content of the page
            
        Returns:
            Dictionary with accessibility metrics and score (0-100)
        """
        logger.info("Analyzing accessibility")
        
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Color contrast analysis
        contrast_score = 0
        foreground_colors = []
        background_colors = []
        
        # Extract text colors
        for tag in soup.find_all(style=True):
            style = tag['style']
            color_match = re.search(r'color:\s*(#[0-9a-fA-F]{3,6}|rgba?\([^)]+\))', style)
            if color_match:
                foreground_colors.append(color_match.group(1))
            
            bg_match = re.search(r'background(-color)?:\s*(#[0-9a-fA-F]{3,6}|rgba?\([^)]+\))', style)
            if bg_match and bg_match.group(2):
                background_colors.append(bg_match.group(2))
        
        # Check for contrast issues
        contrast_issues = 0
        contrast_checks = 0
        
        for fg in foreground_colors:
            for bg in background_colors:
                try:
                    # Convert colors to RGB if needed
                    fg_rgb = self._color_to_rgb(fg)
                    bg_rgb = self._color_to_rgb(bg)
                    
                    if fg_rgb and bg_rgb:
                        # Calculate contrast ratio
                        ratio = wcag_contrast_ratio.rgb(fg_rgb, bg_rgb)
                        contrast_checks += 1
                        
                        # WCAG AA requires 4.5:1 for normal text
                        if ratio < 4.5:
                            contrast_issues += 1
                except:
                    continue
        
        # Calculate contrast score
        if contrast_checks > 0:
            contrast_score = 100 - (contrast_issues / contrast_checks * 100)
        else:
            contrast_score = 50  # Default if no checks performed
        
        # Alt text analysis
        images = soup.find_all('img')
        images_with_alt = [img for img in images if img.get('alt')]
        
        # Calculate alt text score
        alt_text_score = 0
        if images:
            alt_text_ratio = len(images_with_alt) / len(images)
            alt_text_score = alt_text_ratio * 100
        else:
            alt_text_score = 100  # No images, no alt text needed
        
        # ARIA attributes analysis
        aria_score = 0
        elements_with_aria = soup.find_all(attrs=lambda attr: any(a.startswith('aria-') for a in attr if isinstance(a, str)))
        
        # Check for landmark roles
        has_main = bool(soup.find('main')) or bool(soup.find(attrs={'role': 'main'}))
        has_nav = bool(soup.find('nav')) or bool(soup.find(attrs={'role': 'navigation'}))
        has_header = bool(soup.find('header')) or bool(soup.find(attrs={'role': 'banner'}))
        has_footer = bool(soup.find('footer')) or bool(soup.find(attrs={'role': 'contentinfo'}))
        
        # Calculate ARIA score
        landmarks_score = 0
        if has_main:
            landmarks_score += 25
        if has_nav:
            landmarks_score += 25
        if has_header:
            landmarks_score += 25
        if has_footer:
            landmarks_score += 25
            
        aria_score = landmarks_score
        if elements_with_aria:
            aria_score = (aria_score + 50) / 2  # Average with landmark score
        
        # Keyboard navigation analysis
        keyboard_score = 0
        
        # Check for tabindex
        elements_with_tabindex = soup.find_all(attrs={'tabindex': True})
        
        # Check for focus styles
        has_focus_styles = ':focus' in html_content or 'focus' in html_content
        
        # Check for skip links
        has_skip_link = bool(soup.find('a', string=re.compile(r'skip|jump to', re.I)))
        
        # Calculate keyboard score
        keyboard_score = 0
        if elements_with_tabindex:
            keyboard_score += 30
        if has_focus_styles:
            keyboard_score += 40
        if has_skip_link:
            keyboard_score += 30
        
        # Form labels analysis
        form_labels_score = 0
        inputs = soup.find_all(['input', 'select', 'textarea'])
        inputs_with_labels = 0
        
        for input_field in inputs:
            if input_field.get('id'):
                label = soup.find('label', {'for': input_field['id']})
                if label:
                    inputs_with_labels += 1
            elif input_field.get('aria-label') or input_field.get('aria-labelledby'):
                inputs_with_labels += 1
        
        # Calculate form labels score
        if inputs:
            labels_ratio = inputs_with_labels / len(inputs)
            form_labels_score = labels_ratio * 100
        else:
            form_labels_score = 100  # No inputs, no labels needed
        
        # Calculate overall accessibility score
        accessibility_score = (
            (self.accessibility_weights['color_contrast'] * contrast_score) +
            (self.accessibility_weights['alt_text'] * alt_text_score) +
            (self.accessibility_weights['aria_attributes'] * aria_score) +
            (self.accessibility_weights['keyboard_nav'] * keyboard_score) +
            (self.accessibility_weights['form_labels'] * form_labels_score)
        )
        
        return {
            'color_contrast': {
                'contrast_issues': contrast_issues,
                'contrast_checks': contrast_checks,
                'score': contrast_score
            },
            'alt_text': {
                'images_total': len(images),
                'images_with_alt': len(images_with_alt),
                'score': alt_text_score
            },
            'aria_attributes': {
                'elements_with_aria': len(elements_with_aria),
                'landmarks': {
                    'has_main': has_main,
                    'has_nav': has_nav,
                    'has_header': has_header,
                    'has_footer': has_footer
                },
                'score': aria_score
            },
            'keyboard_nav': {
                'elements_with_tabindex': len(elements_with_tabindex),
                'has_focus_styles': has_focus_styles,
                'has_skip_link': has_skip_link,
                'score': keyboard_score
            },
            'form_labels': {
                'inputs_total': len(inputs),
                'inputs_with_labels': inputs_with_labels,
                'score': form_labels_score
            },
            'accessibility_score': accessibility_score
        }
        
    def _color_to_rgb(self, color):
        """Convert various color formats to RGB tuple."""
        if not color:
            return None
            
        if color.startswith('#'):
            # Hex color
            if len(color) == 4:  # #RGB format
                r = int(color[1] + color[1], 16) / 255
                g = int(color[2] + color[2], 16) / 255
                b = int(color[3] + color[3], 16) / 255
                return (r, g, b)
            elif len(color) == 7:  # #RRGGBB format
                r = int(color[1:3], 16) / 255
                g = int(color[3:5], 16) / 255
                b = int(color[5:7], 16) / 255
                return (r, g, b)
        elif color.startswith('rgb'):
            # RGB or RGBA color
            values = re.findall(r'\d+', color)
            if len(values) >= 3:
                r = int(values[0]) / 255
                g = int(values[1]) / 255
                b = int(values[2]) / 255
                return (r, g, b)
        
        return None
    
    def analyze_visual_design(self, html_content):
        """
        Analyze visual design aspects of the webpage.
        
        Args:
            html_content: HTML content of the page
            
        Returns:
            Dictionary with visual design metrics and score (0-100)
        """
        logger.info("Analyzing visual design")
        
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Extract all style attributes and CSS
        styles = []
        for tag in soup.find_all(style=True):
            if tag.get('style'):
                styles.append(tag['style'])
            
        # Extract all colors used
        colors = []
        # Iterate through each style and extract colors
        for style in styles:
            color_matches = re.findall(r'color:\s*(#[0-9a-fA-F]{3,6}|rgba?\([^)]+\))', style)
            if color_matches:
                colors.extend(color_matches)
            
            bg_matches = re.findall(r'background(-color)?:\s*(#[0-9a-fA-F]{3,6}|rgba?\([^)]+\))', style)
            if bg_matches:
                colors.extend([m[1] for m in bg_matches if len(m) > 1])
        
        # Convert hex to RGB if needed and normalize
        normalized_colors = []
        for color in colors:
            try:
                if not color or not isinstance(color, str):
                    continue
                    
                if color.startswith('#'):
                    c = Color(color)
                    normalized_colors.append(c)
                elif color.startswith('rgb'):
                    # Extract RGB values
                    rgb_values = re.findall(r'\d+', color)
                    if len(rgb_values) >= 3:
                        r, g, b = [int(v)/255 for v in rgb_values[:3]]
                        c = Color(rgb=(r, g, b))
                        normalized_colors.append(c)
            except:
                continue
        
        # Color harmony analysis
        color_harmony_score = 0
        if normalized_colors:
            # Check for complementary colors
            if len(normalized_colors) >= 2:
                color_harmony_score = 70  # Base score for having multiple colors
                
                # Check for too many colors (could be overwhelming)
                if len(set(normalized_colors)) > 5:
                    color_harmony_score -= 20
            else:
                color_harmony_score = 50  # Single color scheme
        
        # Visual hierarchy analysis
        headings = {
            'h1': len(soup.find_all('h1')),
            'h2': len(soup.find_all('h2')),
            'h3': len(soup.find_all('h3')),
            'h4': len(soup.find_all('h4')),
            'h5': len(soup.find_all('h5')),
            'h6': len(soup.find_all('h6'))
        }
        
        # Check proper heading hierarchy
        hierarchy_score = 0
        if headings['h1'] > 0:
            hierarchy_score += 30  # Has H1
            
            # Check if H1 comes before H2, etc.
            heading_order_correct = True
            all_headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            current_level = 1
            
            for heading in all_headings:
                heading_level = int(heading.name[1])
                if heading_level > current_level + 1:
                    heading_order_correct = False
                    break
                current_level = heading_level
            
            if heading_order_correct:
                hierarchy_score += 40
            
            # Check if there's only one H1
            if headings['h1'] == 1:
                hierarchy_score += 30
            else:
                hierarchy_score -= 10 * (headings['h1'] - 1)  # Penalty for multiple H1s
        
        # Whitespace analysis
        whitespace_score = 0
        paragraphs = soup.find_all('p')
        if paragraphs:
            # Check if paragraphs have margin/padding
            whitespace_score = 50  # Base score
            
            # Look for margin/padding in styles
            margin_padding_count = 0
            for style in styles:
                if 'margin' in style or 'padding' in style:
                    margin_padding_count += 1
            
            whitespace_score += min(50, margin_padding_count * 5)
        
        # Typography analysis
        typography_score = 0
        font_families = set()
        for style in styles:
            font_matches = re.findall(r'font-family:\s*([^;]+)', style)
            for match in font_matches:
                families = [f.strip().strip("'").strip('"') for f in match.split(',')]
                font_families.update(families)
        
        # Check number of font families (too many is bad)
        if font_families:
            if len(font_families) <= 3:
                typography_score = 100  # Good: 1-3 font families
            elif len(font_families) <= 5:
                typography_score = 70   # OK: 4-5 font families
            else:
                typography_score = 40   # Poor: >5 font families
        else:
            typography_score = 60  # Default fonts, not specified
        
        # Design consistency
        consistency_score = 0
        
        # Check button consistency
        buttons = soup.find_all(['button', 'a'], class_=re.compile(r'btn|button', re.I))
        if buttons:
            button_classes = [' '.join(btn.get('class', [])) for btn in buttons]
            unique_button_classes = set(button_classes)
            
            # If most buttons share the same classes, that's good consistency
            if len(unique_button_classes) <= 3:  # Allow for primary, secondary, tertiary
                consistency_score += 50
            else:
                consistency_score += 30
        
        # Check header/footer consistency
        header = soup.find('header')
        footer = soup.find('footer')
        if header and footer:
            consistency_score += 50
        elif header or footer:
            consistency_score += 25
        
        # Calculate overall visual design score
        visual_design_score = (
            (self.visual_design_weights['color_harmony'] * color_harmony_score) +
            (self.visual_design_weights['visual_hierarchy'] * hierarchy_score) +
            (self.visual_design_weights['consistency'] * consistency_score) +
            (self.visual_design_weights['whitespace'] * whitespace_score) +
            (self.visual_design_weights['typography'] * typography_score)
        )
        
        return {
            'color_harmony': {
                'unique_colors': len(set(normalized_colors)),
                'score': color_harmony_score
            },
            'visual_hierarchy': {
                'heading_structure': headings,
                'score': hierarchy_score
            },
            'consistency': {
                'has_header': bool(header),
                'has_footer': bool(footer),
                'score': consistency_score
            },
            'whitespace': {
                'score': whitespace_score
            },
            'typography': {
                'font_families': list(font_families),
                'score': typography_score
            },
            'visual_design_score': visual_design_score
        }
        
    def analyze_advanced_ui(self, url):
        """
        Perform advanced UI testing using headless browser automation.
        Note: This requires a browser driver to be installed.
        
        Args:
            url: URL of the page to analyze
            
        Returns:
            Dictionary with advanced UI metrics and score (0-100)
        """
        logger.info(f"Performing advanced UI testing for {url}")
        
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.common.exceptions import TimeoutException, WebDriverException
            import time
            
            # Set up headless Chrome
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            # Initialize the driver
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            
            # Set page load timeout
            driver.set_page_load_timeout(30)
            
            try:
                # Load the page
                driver.get(url)
                
                # Wait for page to load
                time.sleep(3)
                
                # Get page performance metrics
                navigation_timing = driver.execute_script("""
                    var performance = window.performance;
                    var timings = performance.timing;
                    return {
                        loadTime: timings.loadEventEnd - timings.navigationStart,
                        domContentLoaded: timings.domContentLoadedEventEnd - timings.navigationStart,
                        firstPaint: performance.getEntriesByType('paint')[0] ? 
                                   performance.getEntriesByType('paint')[0].startTime : 0,
                        firstContentfulPaint: performance.getEntriesByType('paint')[1] ? 
                                            performance.getEntriesByType('paint')[1].startTime : 0
                    };
                """)
                
                # Check for layout shifts
                layout_shifts = driver.execute_script("""
                    var shifts = 0;
                    if (typeof PerformanceObserver !== 'undefined') {
                        try {
                            var observer = new PerformanceObserver(function(list) {
                                shifts += list.getEntries().length;
                            });
                            observer.observe({type: 'layout-shift', buffered: true});
                            return shifts;
                        } catch (e) {
                            return 0;
                        }
                    }
                    return 0;
                """)
                
                # Test responsive behavior
                responsive_results = {}
                for width in [375, 768, 1024, 1440]:  # Mobile, tablet, laptop, desktop
                    driver.set_window_size(width, 800)
                    time.sleep(1)  # Allow layout to adjust
                    
                    # Check if elements are visible and properly sized
                    responsive_results[width] = driver.execute_script("""
                        var results = {};
                        // Check if main content is visible
                        var mainContent = document.querySelector('main') || 
                                         document.querySelector('#content') || 
                                         document.querySelector('.content');
                        results.mainContentVisible = mainContent ? 
                            mainContent.getBoundingClientRect().width > 0 : false;
                        
                        // Check for horizontal scrollbar
                        results.hasHorizontalScroll = 
                            document.body.scrollWidth > document.body.clientWidth;
                        
                        // Check if navigation is adapted
                        var nav = document.querySelector('nav') || 
                                 document.querySelector('.navbar') || 
                                 document.querySelector('#navigation');
                        results.navVisible = nav ? 
                            nav.getBoundingClientRect().width > 0 : false;
                        
                        return results;
                    """)
                
                # Take screenshots for visual analysis
                screenshot = driver.get_screenshot_as_base64()
                
                # Calculate scores
                performance_score = 0
                if navigation_timing['loadTime'] < 3000:  # Less than 3 seconds
                    performance_score = 100
                elif navigation_timing['loadTime'] < 5000:  # Less than 5 seconds
                    performance_score = 80
                elif navigation_timing['loadTime'] < 8000:  # Less than 8 seconds
                    performance_score = 60
                else:
                    performance_score = 40
                
                # Calculate responsive score
                responsive_score = 0
                responsive_checks = 0
                responsive_passes = 0
                
                for width, results in responsive_results.items():
                    responsive_checks += 2  # mainContentVisible and hasHorizontalScroll
                    
                    if results.get('mainContentVisible', False):
                        responsive_passes += 1
                    
                    if not results.get('hasHorizontalScroll', True):  # No horizontal scroll is good
                        responsive_passes += 1
                
                responsive_score = (responsive_passes / responsive_checks) * 100 if responsive_checks > 0 else 0
                
                # Calculate overall advanced UI score
                advanced_ui_score = (performance_score * 0.6) + (responsive_score * 0.4)
                
                driver.quit()
                
                return {
                    'performance': {
                        'load_time': navigation_timing['loadTime'],
                        'dom_content_loaded': navigation_timing['domContentLoaded'],
                        'first_paint': navigation_timing['firstPaint'],
                        'first_contentful_paint': navigation_timing['firstContentfulPaint'],
                        'score': performance_score
                    },
                    'responsive_design': {
                        'breakpoints_tested': list(responsive_results.keys()),
                        'horizontal_scroll_issues': any(r.get('hasHorizontalScroll', False) for r in responsive_results.values()),
                        'score': responsive_score
                    },
                    'layout_shifts': layout_shifts,
                    'advanced_ui_score': advanced_ui_score
                }
                
            except (TimeoutException, WebDriverException) as e:
                logger.error(f"Error during advanced UI testing: {str(e)}")
                driver.quit()
                return {
                    'error': str(e),
                    'advanced_ui_score': 0
                }
                
        except ImportError as e:
            logger.warning(f"Selenium not available for advanced UI testing: {str(e)}")
            return {
                'error': 'Selenium not available for advanced UI testing',
                'advanced_ui_score': 0
            }
    
    def analyze_ui_ux(self, url, html_content, run_advanced_tests=False):
        """
        Perform comprehensive UI/UX analysis combining all analysis methods.
        
        Args:
            url: URL of the page to analyze
            html_content: HTML content of the page
            run_advanced_tests: Whether to run advanced browser-based tests
            
        Returns:
            Dictionary with comprehensive UI/UX analysis results
        """
        logger.info(f"Starting comprehensive UI/UX analysis for {url}")
        
        # Validate HTML content
        if not html_content or not isinstance(html_content, str):
            raise ValueError("Invalid HTML content provided for UI/UX analysis")
            
        try:
            # Perform basic analyses with error handling
            try:
                logger.info("Analyzing visual design")
                visual_design_results = self.analyze_visual_design(html_content)
            except Exception as e:
                logger.error(f"Error in visual design analysis: {str(e)}")
                visual_design_results = {'error': str(e), 'visual_design_score': 0}
                
            try:
                logger.info("Analyzing interaction design")
                interaction_design_results = self.analyze_interaction_design(html_content)
            except Exception as e:
                logger.error(f"Error in interaction design analysis: {str(e)}")
                interaction_design_results = {'error': str(e), 'interaction_design_score': 0}
                
            try:
                logger.info("Analyzing accessibility")
                accessibility_results = self.analyze_accessibility(html_content)
            except Exception as e:
                logger.error(f"Error in accessibility analysis: {str(e)}")
                accessibility_results = {'error': str(e), 'accessibility_score': 0}
            
            # Perform advanced tests if requested
            advanced_ui_results = None
            if run_advanced_tests:
                try:
                    advanced_ui_results = self.run_advanced_ui_tests(url)
                except Exception as e:
                    logger.error(f"Error running advanced UI tests: {str(e)}")
                    advanced_ui_results = {
                        'error': str(e),
                        'advanced_ui_score': 0
                    }
            
            # Calculate overall UI/UX score with safe defaults
            visual_score = visual_design_results.get('visual_design_score', 0)
            interaction_score = interaction_design_results.get('interaction_design_score', 0)
            accessibility_score = accessibility_results.get('accessibility_score', 0)
            
            # Weight the scores
            ui_ux_score = (
                (0.4 * visual_score) +
                (0.3 * interaction_score) +
                (0.3 * accessibility_score)
            )
            
            # If advanced tests were run, include them in the score
            if advanced_ui_results:
                advanced_score = advanced_ui_results.get('advanced_ui_score', 0)
                ui_ux_score = (0.7 * ui_ux_score) + (0.3 * advanced_score)
            
            # Determine rating
            if ui_ux_score >= 90:
                ui_ux_rating = "Excellent"
            elif ui_ux_score >= 80:
                ui_ux_rating = "Very Good"
            elif ui_ux_score >= 70:
                ui_ux_rating = "Good"
            elif ui_ux_score >= 60:
                ui_ux_rating = "Fair"
            elif ui_ux_score >= 50:
                ui_ux_rating = "Needs Improvement"
            else:
                ui_ux_rating = "Poor"
            
            # Generate recommendations
            recommendations = []
            
            # Visual design recommendations
            if visual_score < 70:
                if visual_design_results.get('color_harmony_score', 0) < 60:
                    recommendations.append("Improve color harmony and contrast for better visual appeal")
                if visual_design_results.get('layout_balance_score', 0) < 60:
                    recommendations.append("Enhance layout balance and whitespace usage")
                if visual_design_results.get('typography_score', 0) < 60:
                    recommendations.append("Improve typography consistency and readability")
            
            # Interaction design recommendations
            if interaction_score < 70:
                if interaction_design_results.get('navigation_score', 0) < 60:
                    recommendations.append("Simplify navigation structure for better user flow")
                if interaction_design_results.get('form_design_score', 0) < 60:
                    recommendations.append("Enhance form design with better validation and user feedback")
                if interaction_design_results.get('call_to_action_score', 0) < 60:
                    recommendations.append("Make call-to-action elements more prominent and clear")
            
            # Accessibility recommendations
            if accessibility_score < 70:
                if accessibility_results.get('alt_text_score', 0) < 60:
                    recommendations.append("Add descriptive alt text to all images")
                if accessibility_results.get('aria_score', 0) < 60:
                    recommendations.append("Implement ARIA attributes for better screen reader support")
                if accessibility_results.get('keyboard_nav_score', 0) < 60:
                    recommendations.append("Improve keyboard navigation support")
            
            # Advanced UI recommendations
            if advanced_ui_results and advanced_ui_results.get('advanced_ui_score', 0) < 70:
                if advanced_ui_results.get('responsive_score', 0) < 60:
                    recommendations.append("Fix responsive design issues across different screen sizes")
        
            return {
                'url': url,
                'visual_design': visual_design_results,
                'interaction_design': interaction_design_results,
                'accessibility': accessibility_results,
                'advanced_ui': advanced_ui_results,
                'ui_ux_score': ui_ux_score,
                'ui_ux_rating': ui_ux_rating,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in UI/UX analysis: {str(e)}")
            return {
                'url': url,
                'error': f"Error in UI/UX analysis: {str(e)}",
                'ui_ux_score': 0,
                'ui_ux_rating': 'Error',
                'recommendations': ["Fix technical issues with the website to enable proper analysis"]
            }

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file, send_from_directory
from werkzeug.utils import secure_filename
from crawler import WebCrawler, Crawler
from analyzer import SEOAnalyzer
from ui_ux_integration import EnhancedSEOAnalyzer
from indexer import Indexer
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['SECRET_KEY'] = os.urandom(24)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
crawler = Crawler()
indexer = Indexer()
analyzer = SEOAnalyzer()
enhanced_analyzer = EnhancedSEOAnalyzer()


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/ui-ux-analysis')
def ui_ux_analysis_page():
    """Render the UI/UX analysis page"""
    return render_template('ui_ux_analysis.html')


@app.route('/analyze-ui-ux', methods=['POST'])
def analyze_ui_ux():
    """Analyze UI/UX aspects of a URL and return detailed scores"""
    data = request.form
    url = data.get('url')
    run_advanced_tests = data.get('run_advanced_tests', 'false').lower() == 'true'
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    try:
        # Fetch HTML content
        html_content = enhanced_analyzer.fetch_url(url)
        
        # Perform enhanced UI/UX analysis
        results = enhanced_analyzer.ui_ux_analyzer.analyze_ui_ux(url, html_content, run_advanced_tests)
        
        # Store results in session for visualization
        session_id = str(uuid.uuid4())
        session[f'ui_ux_results_{session_id}'] = results
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'ui_ux_score': results['ui_ux_score'],
            'ui_ux_rating': results['ui_ux_rating'],
            'recommendations': results['recommendations']
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({
            'error': str(e),
            'details': error_details
        }), 500


@app.route('/enhanced-seo-analysis', methods=['POST'])
def enhanced_seo_analysis():
    """Perform enhanced SEO analysis including advanced UI/UX metrics"""
    data = request.form
    url = data.get('url')
    query = data.get('query', None)
    max_depth = int(data.get('depth', 0))
    run_advanced_tests = data.get('run_advanced_tests', 'false').lower() == 'true'
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    try:
        # Perform enhanced SEO analysis
        results = enhanced_analyzer.perform_enhanced_analysis(
            url,
            query=query,
            max_depth=max_depth,
            run_advanced_tests=run_advanced_tests
        )
        
        # Store results in session for visualization
        session_id = str(uuid.uuid4())
        session[f'enhanced_seo_results_{session_id}'] = results
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'seo_score': results['seo_score'],
            'ui_ux_score': results['user_experience']['combined_user_experience_score'],
            'recommendations': results['recommendations']
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({
            'error': str(e),
            'details': error_details
        }), 500


@app.route('/analyze', methods=['POST'])
def analyze_url():
    """Analyze a URL and return SEO score"""
    data = request.form
    url = data.get('url')
    query = data.get('query', None)
    max_depth = int(data.get('depth', 0))
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    try:
        logger.info(f"Starting analysis for URL: {url}, query: {query}, max_depth: {max_depth}")
        
        # Analyze URL
        results = analyzer.analyze_url(
            url, 
            query=query, 
            max_depth=max_depth
        )
        
        # Store results in session for visualization
        session_id = os.urandom(16).hex()
        
        # Ensure uploads directory exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            
        results_file = os.path.join(app.config['UPLOAD_FOLDER'], f'results_{session_id}.json')
        
        logger.info(f"Saving results to {results_file}")
        with open(results_file, 'w') as f:
            json.dump(results, f)
        
        # Return results and session ID
        return jsonify({
            'success': True,
            'session_id': session_id,
            'results': results
        })
    
    except Exception as e:
        logger.error(f"Error analyzing URL {url}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/results/<session_id>')
def view_results(session_id):
    """View detailed results for a session"""
    results_file = os.path.join(app.config['UPLOAD_FOLDER'], f'results_{session_id}.json')
    
    if not os.path.exists(results_file):
        return render_template('error.html', message='Results not found')
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Generate visualizations
    factor_scores = results['factor_scores']
    factor_weights = results['factor_weights']
    
    # Create radar chart data
    categories = list(factor_scores.keys())
    categories = [cat.replace('_score', '') for cat in categories]
    values = list(factor_scores.values())
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='SEO Factors'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title='SEO Factor Scores',
        showlegend=False
    )
    
    radar_chart = fig.to_html(full_html=False)
    
    # Create pie chart for factor weights
    weight_labels = list(factor_weights.keys())
    weight_values = list(factor_weights.values())
    
    fig_pie = px.pie(
        values=weight_values,
        names=weight_labels,
        title='SEO Factor Weights',
        hole=0.3
    )
    
    pie_chart = fig_pie.to_html(full_html=False)
    
    # Create bar chart for factor scores
    fig_bar = px.bar(
        x=categories,
        y=values,
        title='SEO Factor Scores',
        labels={'x': 'Factor', 'y': 'Score'},
        color=values,
        color_continuous_scale='Viridis'
    )
    
    bar_chart = fig_bar.to_html(full_html=False)
    
    return render_template(
        'results.html',
        results=results,
        radar_chart=radar_chart,
        pie_chart=pie_chart,
        bar_chart=bar_chart
    )


@app.route('/batch', methods=['GET', 'POST'])
def batch_analyze():
    """Batch analyze multiple URLs"""
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('batch.html', error='No file uploaded')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('batch.html', error='No file selected')
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Read CSV file
                df = pd.read_csv(filepath)
                
                if 'url' not in df.columns:
                    return render_template('batch.html', error='CSV must contain a "url" column')
                
                # Analyze each URL
                results = []
                for _, row in df.iterrows():
                    url = row['url']
                    query = row.get('query', None)
                    
                    try:
                        result = analyzer.analyze_url(url, query=query, max_depth=0)
                        result['url'] = url
                        results.append(result)
                    except Exception as e:
                        results.append({
                            'url': url,
                            'error': str(e),
                            'seo_score': 0
                        })
                
                # Save results
                session_id = os.urandom(16).hex()
                results_file = os.path.join(app.config['UPLOAD_FOLDER'], f'batch_{session_id}.json')
                
                with open(results_file, 'w') as f:
                    json.dump(results, f)
                
                return render_template('batch_results.html', results=results, session_id=session_id)
            
            except Exception as e:
                return render_template('batch.html', error=str(e))
        else:
            return render_template('batch.html', error='File must be a CSV')
    
    return render_template('batch.html')


@app.route('/compare', methods=['GET', 'POST'])
def compare_urls():
    """Compare SEO scores of multiple URLs"""
    if request.method == 'POST':
        urls = request.form.get('urls', '').strip().split('\n')
        query = request.form.get('query', None)
        
        if not urls or not urls[0]:
            return render_template('compare.html', error='No URLs provided')
        
        # Analyze each URL
        results = []
        for url in urls:
            url = url.strip()
            if url:
                try:
                    result = analyzer.analyze_url(url, query=query, max_depth=0)
                    result['url'] = url
                    results.append(result)
                except Exception as e:
                    results.append({
                        'url': url,
                        'error': str(e),
                        'seo_score': 0
                    })
        
        # Create comparison chart
        urls = [r['url'] for r in results]
        scores = [r.get('seo_score', 0) for r in results]
        
        fig = px.bar(
            x=urls,
            y=scores,
            title='SEO Score Comparison',
            labels={'x': 'URL', 'y': 'SEO Score'},
            color=scores,
            color_continuous_scale='Viridis'
        )
        
        comparison_chart = fig.to_html(full_html=False)
        
        # Create radar chart for factor comparison
        fig_radar = go.Figure()
        
        for i, result in enumerate(results):
            if 'factor_scores' in result:
                categories = list(result['factor_scores'].keys())
                categories = [cat.replace('_score', '') for cat in categories]
                values = list(result['factor_scores'].values())
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=f"URL {i+1}"
                ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title='SEO Factor Comparison'
        )
        
        radar_comparison = fig_radar.to_html(full_html=False)
        
        return render_template(
            'compare_results.html',
            results=results,
            comparison_chart=comparison_chart,
            radar_comparison=radar_comparison
        )
    
    return render_template('compare.html')


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for analyzing a URL"""
    data = request.json
    
    if not data or 'url' not in data:
        return jsonify({'error': 'URL is required'}), 400
    
    url = data['url']
    query = data.get('query', None)
    max_depth = int(data.get('depth', 0))
    
    try:
        # Analyze URL
        results = analyzer.analyze_url(
            url, 
            query=query, 
            max_depth=max_depth
        )
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Route for URL comparison form
@app.route('/compare', methods=['GET', 'POST'])
def compare():
    """Handle URL comparison form"""
    if request.method == 'POST':
        try:
            # Get URLs from form
            urls_text = request.form.get('urls', '')
            query = request.form.get('query', None)
            
            # Split URLs by newline and filter empty lines
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            
            if not urls:
                return render_template('compare.html', error='Please enter at least one URL')
                
            if len(urls) > 5:
                return render_template('compare.html', error='Maximum 5 URLs allowed for comparison')
            
            # Create analyzer instance
            analyzer = SEOAnalyzer()
            
            # Compare URLs
            results = analyzer.compare_urls(urls, query=query, max_depth=0)
            
            # Generate radar chart for comparison
            import plotly.graph_objects as go
            import plotly.express as px
            
            # Extract factor names and scores for each URL
            factor_names = ['Content Quality', 'Backlink Authority', 'Technical SEO', 
                           'User Experience', 'Search Intent', 'Page Speed', 'Brand & Social']
            
            # Create radar chart
            fig = go.Figure()
            
            for i, result in enumerate(results):
                if 'error' in result:
                    continue
                    
                # Extract scores
                scores = [
                    result['factor_scores']['content_quality_score'],
                    result['factor_scores']['backlink_authority_score'],
                    result['factor_scores']['technical_seo_score'],
                    result['factor_scores']['user_experience_score'],
                    result['factor_scores']['search_intent_score'],
                    result['factor_scores']['page_speed_score'],
                    result['factor_scores']['brand_social_score']
                ]
                
                # Add trace for this URL
                fig.add_trace(go.Scatterpolar(
                    r=scores,
                    theta=factor_names,
                    fill='toself',
                    name=result['url'][:30]  # Truncate URL for legend
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True
            )
            
            # Convert to HTML
            radar_comparison = fig.to_html(full_html=False)
            
            return render_template('compare_results.html', 
                                  results=results, 
                                  radar_comparison=radar_comparison,
                                  query=query)
                                  
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logging.error(f"Error in URL comparison: {str(e)}\n{error_details}")
            return render_template('compare.html', error=f"Error: {str(e)}")
    
    # GET request - show form
    return render_template('compare.html')

# Add results route to display analysis results
@app.route('/results/<session_id>')
def show_results(session_id):
    """Show analysis results"""
    try:
        results_file = os.path.join(app.config['UPLOAD_FOLDER'], f'results_{session_id}.json')
        
        if not os.path.exists(results_file):
            return render_template('error.html', error='Results not found'), 404
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Generate charts
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import pandas as pd
            
            # Extract factor scores
            factor_scores = {}
            for factor, score in results['factor_scores'].items():
                factor_scores[factor.replace('_', ' ').title()] = score
            
            # Create radar chart
            categories = list(factor_scores.keys())
            values = list(factor_scores.values())
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=results['url']
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="SEO Factor Scores"
            )
            radar_chart = fig.to_html(full_html=False)
            
            # Create pie chart for factor weights
            weights_df = pd.DataFrame({
                'Factor': categories,
                'Weight': [0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1]  # Example weights
            })
            
            pie_fig = px.pie(weights_df, values='Weight', names='Factor', title='SEO Factor Weights')
            pie_chart = pie_fig.to_html(full_html=False)
            
            # Create bar chart
            bar_fig = px.bar(x=categories, y=values, title="SEO Factor Scores")
            bar_chart = bar_fig.to_html(full_html=False)
            
        except Exception as chart_error:
            logger.error(f"Error generating charts: {str(chart_error)}")
            radar_chart = "<div class='alert alert-warning'>Error generating radar chart</div>"
            pie_chart = "<div class='alert alert-warning'>Error generating pie chart</div>"
            bar_chart = "<div class='alert alert-warning'>Error generating bar chart</div>"
        
        return render_template('results.html', results=results, session_id=session_id,
                              radar_chart=radar_chart, pie_chart=pie_chart, bar_chart=bar_chart)
    
    except Exception as e:
        logger.error(f"Error showing results for session {session_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('error.html', error=str(e)), 500

# Add detailed error logging
import traceback
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Modify the analyze_url function to include better error handling
def log_exception(sender, exception):
    logger.error(f"Unhandled exception: {exception}\n{traceback.format_exc()}")

from flask import got_request_exception
got_request_exception.connect(log_exception, app)

if __name__ == '__main__':
    # Make sure uploads directory exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    app.run(host='0.0.0.0', port=5003, debug=True)

import os
import json
import time
import uuid
import numpy as np
import chainlit as cl
from crawler import WebCrawler, Crawler
from analyzer import SEOAnalyzer
from ui_ux_integration import EnhancedSEOAnalyzer
from indexer import Indexer
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Initialize components
crawler = Crawler()
indexer = Indexer()
analyzer = SEOAnalyzer()
enhanced_analyzer = EnhancedSEOAnalyzer()

# Ensure upload directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@cl.on_chat_start
def start():
    cl.user_session.set("session_id", str(uuid.uuid4()))
    
    # Welcome message with instructions
    welcome_message = """
    # Welcome to SEO Analyzer
    
    This tool helps you analyze websites for SEO performance using advanced algorithms.
    
    ## Available Commands:
    - `/analyze [url] [query] [depth]` - Analyze a single URL
    - `/enhanced [url] [query] [depth]` - Run enhanced analysis with UI/UX metrics
    - `/compare [url1] [url2] ...` - Compare multiple URLs
    - `/help` - Show this help message
    
    Example: `/analyze https://example.com best seo practices 0`
    """
    
    cl.Message(content=welcome_message).send()

@cl.on_message
async def main(message: cl.Message):
    msg = message.content.strip()
    session_id = cl.user_session.get("session_id")
    
    # Help command
    if msg.startswith("/help"):
        help_message = """
        # SEO Analyzer Commands
        
        - `/analyze [url] [query] [depth]` - Analyze a single URL
        - `/enhanced [url] [query] [depth]` - Run enhanced analysis with UI/UX metrics
        - `/compare [url1] [url2] ...` - Compare multiple URLs
        - `/help` - Show this help message
        
        Example: `/analyze https://example.com best seo practices 0`
        """
        await cl.Message(content=help_message).send()
        return
    
    # Analyze command
    elif msg.startswith("/analyze"):
        parts = msg.split()
        if len(parts) < 2:
            await cl.Message(content="Please provide a URL to analyze. Example: `/analyze https://example.com`").send()
            return
            
        url = parts[1]
        query = parts[2] if len(parts) > 2 else None
        try:
            max_depth = int(parts[3]) if len(parts) > 3 else 0
        except ValueError:
            max_depth = 0
            
        # Show analyzing message
        analyzing_msg = cl.Message(content=f"Analyzing {url}... This may take a moment.")
        await analyzing_msg.send()
        
        try:
            # Run analysis
            with cl.Step(name="SEO Analysis", type="tool") as step:
                step.input = f"Analyzing {url} with depth {max_depth}"
                results = analyzer.analyze_url(url, query=query, max_depth=max_depth)
                step.output = f"Analysis complete with score: {results['seo_score']}"
            
            # Create visualizations
            factor_scores = results['factor_scores']
            categories = list(factor_scores.keys())
            categories = [cat.replace('_score', '').replace('_', ' ').title() for cat in categories]
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
                title='SEO Factor Scores'
            )
            
            # Save the chart
            chart_path = os.path.join(UPLOAD_FOLDER, f'chart_{session_id}.html')
            fig.write_html(chart_path)
            
            # Format recommendations
            recommendations = "\n## Recommendations\n\n"
            for rec in results.get('recommendations', []):
                recommendations += f"- {rec}\n"
            
            # Send results
            result_message = f"# SEO Analysis Results for {url}\n\n"
            result_message += f"## Overall SEO Score: {results['seo_score']:.1f}/100\n\n"
            result_message += f"## Factor Scores\n\n"
            
            for cat, val in zip(categories, values):
                result_message += f"- {cat}: {val:.1f}/100\n"
                
            result_message += recommendations
            
            # Send the results
            await cl.Message(content=result_message).send()
            
            # Send the chart as an element
            elements = [
                cl.Plotly(name="SEO Factors", figure=fig)
            ]
            await cl.Message(content="SEO Factor Analysis Chart", elements=elements).send()
            
        except Exception as e:
            await cl.Message(content=f"Error analyzing URL: {str(e)}").send()
    
    # Enhanced analysis command
    elif msg.startswith("/enhanced"):
        parts = msg.split()
        if len(parts) < 2:
            await cl.Message(content="Please provide a URL to analyze. Example: `/enhanced https://example.com`").send()
            return
            
        url = parts[1]
        query = parts[2] if len(parts) > 2 else None
        try:
            max_depth = int(parts[3]) if len(parts) > 3 else 0
        except ValueError:
            max_depth = 0
        
        run_advanced_tests = False  # Set to True if you want to run browser-based tests
        
        # Show analyzing message
        analyzing_msg = cl.Message(content=f"Running enhanced analysis on {url}... This may take a moment.")
        await analyzing_msg.send()
        
        try:
            # Run enhanced analysis
            with cl.Step(name="Enhanced SEO Analysis", type="tool") as step:
                step.input = f"Analyzing {url} with enhanced UI/UX metrics"
                results = enhanced_analyzer.perform_enhanced_analysis(
                    url,
                    query=query,
                    max_depth=max_depth,
                    run_advanced_tests=run_advanced_tests
                )
                step.output = f"Analysis complete with SEO score: {results['seo_score']} and UI/UX score: {results['user_experience']['combined_user_experience_score']}"
            
            # Format results
            result_message = f"# Enhanced SEO Analysis Results for {url}\n\n"
            result_message += f"## Overall SEO Score: {results['seo_score']:.1f}/100\n\n"
            result_message += f"## UI/UX Score: {results['user_experience']['combined_user_experience_score']:.1f}/100\n\n"
            
            # Format recommendations
            recommendations = "\n## Recommendations\n\n"
            for rec in results.get('recommendations', []):
                recommendations += f"- {rec}\n"
                
            result_message += recommendations
            
            # Send the results
            await cl.Message(content=result_message).send()
            
        except Exception as e:
            await cl.Message(content=f"Error performing enhanced analysis: {str(e)}").send()
    
    # Compare command
    elif msg.startswith("/compare"):
        parts = msg.split()
        if len(parts) < 3:
            await cl.Message(content="Please provide at least two URLs to compare. Example: `/compare https://example.com https://another.com`").send()
            return
            
        urls = parts[1:]
        query = None  # Could add an option to specify a query
        
        # Show analyzing message
        analyzing_msg = cl.Message(content=f"Comparing {len(urls)} URLs... This may take a moment.")
        await analyzing_msg.send()
        
        try:
            # Analyze each URL
            results = []
            for url in urls:
                with cl.Step(name=f"Analyzing {url}", type="tool") as step:
                    step.input = f"Analyzing {url}"
                    try:
                        result = analyzer.analyze_url(url, query=query, max_depth=0)
                        result['url'] = url
                        results.append(result)
                        step.output = f"Analysis complete with score: {result['seo_score']}"
                    except Exception as e:
                        results.append({
                            'url': url,
                            'error': str(e),
                            'seo_score': 0
                        })
                        step.output = f"Error: {str(e)}"
            
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
            
            # Create radar chart for factor comparison
            fig_radar = go.Figure()
            
            for i, result in enumerate(results):
                if 'factor_scores' in result:
                    categories = list(result['factor_scores'].keys())
                    categories = [cat.replace('_score', '').replace('_', ' ').title() for cat in categories]
                    values = list(result['factor_scores'].values())
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=f"{result['url']}"
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
            
            # Format comparison results
            result_message = f"# SEO Comparison Results\n\n"
            result_message += "## Overall Scores\n\n"
            
            for result in results:
                if 'error' in result:
                    result_message += f"- {result['url']}: Error - {result['error']}\n"
                else:
                    result_message += f"- {result['url']}: {result['seo_score']:.1f}/100\n"
            
            # Send the results
            await cl.Message(content=result_message).send()
            
            # Send the charts as elements
            elements = [
                cl.Plotly(name="Score Comparison", figure=fig),
                cl.Plotly(name="Factor Comparison", figure=fig_radar)
            ]
            await cl.Message(content="Comparison Charts", elements=elements).send()
            
        except Exception as e:
            await cl.Message(content=f"Error comparing URLs: {str(e)}").send()
    
    # Default response for unrecognized commands
    else:
        await cl.Message(content="I don't understand that command. Type `/help` to see available commands.").send()
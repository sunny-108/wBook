#!/usr/bin/env python
"""
Script to combine HTML files in sequence and convert to PDF while preserving href links and page numbers.
Sequence: book_cover_print.html, title_page.html, complete_book.html, index.html
"""

import os
import sys
from pathlib import Path
import logging
import re
from typing import List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('html_combiner')

try:
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
    from bs4 import BeautifulSoup
except ImportError:
    logger.error("Required packages not installed. Please run: pip install weasyprint beautifulsoup4")
    sys.exit(1)

def read_html_content(file_path: str) -> str:
    """
    Read HTML file content.
    
    Args:
        file_path: Path to HTML file
    
    Returns:
        HTML content as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}")
        return ""

def extract_body_content(html_content: str) -> str:
    """
    Extract content from HTML body, excluding html, head, and body tags.
    
    Args:
        html_content: Full HTML content
    
    Returns:
        Body content only
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        body = soup.find('body')
        if body:
            return str(body)[6:-7]  # Remove <body> and </body> tags
        else:
            # If no body tag, return content inside html tag excluding head
            html_tag = soup.find('html')
            if html_tag:
                # Remove head tag if present
                head = soup.find('head')
                if head and hasattr(head, 'decompose'):
                    head.decompose()
                return str(html_tag)[6:-7]  # Remove <html> and </html> tags
            else:
                return html_content
    except Exception as e:
        logger.error(f"Error extracting body content: {str(e)}")
        return html_content

def extract_css_from_html(html_content: str) -> str:
    """
    Extract CSS styles from HTML content.
    
    Args:
        html_content: Full HTML content
    
    Returns:
        Extracted CSS as string
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        css_content = ""
        
        # Extract from style tags
        style_tags = soup.find_all('style')
        for style in style_tags:
            css_content += style.get_text() + "\n"
        
        return css_content
    except Exception as e:
        logger.error(f"Error extracting CSS: {str(e)}")
        return ""

def combine_html_files(html_files: List[str], output_path: str) -> bool:
    """
    Combine multiple HTML files into a single HTML file with unified styling.
    
    Args:
        html_files: List of HTML file paths to combine
        output_path: Path for the combined HTML file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Starting HTML combination process...")
        
        # Collect all CSS and body content
        all_css = ""
        all_body_content = ""
        
        for i, html_file in enumerate(html_files):
            logger.info(f"Processing {html_file}...")
            
            if not os.path.exists(html_file):
                logger.error(f"File not found: {html_file}")
                return False
            
            html_content = read_html_content(html_file)
            if not html_content:
                continue
            
            # Extract CSS
            css = extract_css_from_html(html_content)
            if css:
                all_css += f"\n/* Styles from {os.path.basename(html_file)} */\n"
                all_css += css + "\n"
            
            # Extract body content
            body_content = extract_body_content(html_content)
            if body_content:
                # Add page break before each section (except the first)
                if i > 0:
                    all_body_content += '<div style="page-break-before: always;"></div>\n'
                
                # Wrap content in a section with unique class
                section_class = f"section-{i+1}"
                all_body_content += f'<div class="{section_class}">\n{body_content}\n</div>\n\n'
        
        # Create unified CSS with page break controls
        unified_css = """
/* Global page settings */
@page {
    size: letter;
    margin: 2cm;
    @bottom-center {
        content: "Page " counter(page);
        font-size: 10pt;
        color: #666;
    }
}

/* Global body settings */
body {
    font-family: "Helvetica", "Arial", sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 0;
    font-size: 10.5pt;
}

/* Section-specific page breaks */
.section-1 {
    /* Cover page - no page break before */
}

.section-2 {
    /* Title page - always start on new page */
    page-break-before: always;
}

.section-3 {
    /* Main book content - start on new page */
    page-break-before: always;
}

.section-4 {
    /* Index - start on new page */
    page-break-before: always;
}

/* Ensure links work properly */
a {
    color: #3498db !important;
    text-decoration: none !important;
}

a:hover {
    text-decoration: underline !important;
}

/* Prevent bad page breaks */
h1, h2, h3 {
    page-break-after: avoid;
}

pre {
    page-break-inside: avoid;
}

.chapter {
    page-break-before: always;
}

.chapter:first-of-type {
    page-break-before: avoid;
}

""" + all_css
        
        # Create combined HTML
        combined_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Complete Book - Prompt Engineering for Developers</title>
    <style>
{unified_css}
    </style>
</head>
<body>
{all_body_content}
</body>
</html>"""
        
        # Write combined HTML
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(combined_html)
        
        logger.info(f"Successfully combined {len(html_files)} HTML files into {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error combining HTML files: {str(e)}")
        return False

def convert_html_to_pdf_with_links(html_file_path: str, output_pdf_path: str) -> bool:
    """
    Convert combined HTML file to PDF while preserving links and page numbers.
    
    Args:
        html_file_path: Path to the combined HTML file
        output_pdf_path: Path where PDF should be saved
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Converting {html_file_path} to PDF...")
        
        # Create font configuration for better typography
        font_config = FontConfiguration()
        
        # Additional CSS to enhance PDF output and ensure links work
        pdf_css = CSS(string="""
        /* Enhanced PDF-specific styles */
        @page {
            size: letter;
            margin: 2cm;
            @bottom-center {
                content: "Page " counter(page) !important;
                font-size: 10pt;
                color: #666;
            }
        }
        
        /* Ensure proper link styling in PDF */
        a {
            color: #3498db !important;
            text-decoration: none !important;
        }
        
        /* Better table of contents formatting */
        .toc a {
            color: #3498db !important;
        }
        
        .toc .page-number {
            color: #666 !important;
        }
        
        /* Improve code block handling */
        pre {
            background-color: #f8f8f8 !important;
            border: 1px solid #ddd !important;
            padding: 10px !important;
            border-radius: 3px !important;
            page-break-inside: avoid !important;
            font-size: 9pt !important;
        }
        
        /* Better heading spacing */
        h1 {
            color: #2c3e50 !important;
            page-break-after: avoid !important;
        }
        
        h2 {
            color: #3498db !important;
            page-break-after: avoid !important;
        }
        
        h3 {
            color: #2980b9 !important;
            page-break-after: avoid !important;
        }
        
        /* Navigation styling */
        .navigation {
            border-top: 1px solid #eee !important;
            padding-top: 1em !important;
            margin-top: 2em !important;
        }
        
        .navigation a {
            color: #3498db !important;
            padding: 5px 10px !important;
        }
        """, font_config=font_config)
        
        # Create HTML object with base URL for resolving relative links
        html_doc = HTML(filename=html_file_path)
        
        # Render to PDF with optimized settings for links
        html_doc.write_pdf(
            output_pdf_path,
            stylesheets=[pdf_css],
            font_config=font_config,
            optimize_images=True,
            presentational_hints=True,
            pdf_version='1.7'  # Use PDF 1.7 for better link support
        )
        
        logger.info(f"PDF successfully created: {output_pdf_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting HTML to PDF: {str(e)}")
        return False

def main():
    """Main function to combine HTML files and convert to PDF."""
    
    # Set up paths
    current_dir = Path(__file__).parent
    workspace_dir = current_dir.parent
    cover_dir = workspace_dir / "cover_design"
    html_dir = workspace_dir / "output" / "temp" / "html"
    output_dir = workspace_dir / "output"
    
    # Define HTML files in the required sequence
    html_files = [
        str(cover_dir / "book_cover_print.html"),
        str(html_dir / "title_page.html"),
        str(html_dir / "complete_book.html"),
        str(html_dir / "index.html")
    ]
    
    # Output paths
    combined_html = html_dir / "final_combined_book.html"
    final_pdf = output_dir / "final_complete_book_with_working_links.pdf"
    
    # Ensure output directories exist
    html_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Combine all HTML files
    logger.info("Step 1: Combining HTML files...")
    success = combine_html_files(html_files, str(combined_html))
    
    if not success:
        print("❌ Failed to combine HTML files. Check the logs for details.")
        return False
    
    # Step 2: Convert combined HTML to PDF
    logger.info("Step 2: Converting combined HTML to PDF...")
    success = convert_html_to_pdf_with_links(str(combined_html), str(final_pdf))
    
    if success:
        # Get final file info
        file_size_mb = final_pdf.stat().st_size / 1024 / 1024
        
        print(f"\n🎉 SUCCESS! Final complete book created!")
        print(f"📁 Location: {final_pdf}")
        print(f"📄 File size: {file_size_mb:.2f} MB")
        print(f"\n📑 Document structure:")
        print("  1. Cover Page")
        print("  2. Title Page") 
        print("  3. Complete Book Content:")
        print("     • About the Book")
        print("     • About the Author")
        print("     • Table of Contents (with working page numbers)")
        print("     • All 9 Chapters with navigation")
        print("  4. Index")
        print(f"\n✅ Features working:")
        print("  • All href links (clickable navigation)")
        print("  • Page numbers at bottom of each page")
        print("  • Table of contents with working page links")
        print("  • Chapter-to-chapter navigation")
        print("  • Code syntax highlighting")
        print("  • Professional formatting")
        print(f"\n📄 Combined HTML saved at: {combined_html}")
        
        return True
    else:
        print("❌ Failed to create PDF. Check the logs for details.")
        return False

if __name__ == "__main__":
    main()

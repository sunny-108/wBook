#!/usr/bin/env python
"""
Script to convert the complete HTML book file to PDF while preserving href links and page numbers.
"""

import os
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('html2pdf')

try:
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
except ImportError:
    logger.error("WeasyPrint not installed. Please run: pip install weasyprint")
    sys.exit(1)

def convert_html_to_pdf(html_file_path: str, output_pdf_path: str):
    """
    Convert HTML file to PDF with preserved links and page numbers.
    
    Args:
        html_file_path: Path to the HTML file
        output_pdf_path: Path where PDF should be saved
    """
    try:
        # Check if HTML file exists
        if not os.path.exists(html_file_path):
            logger.error(f"HTML file not found: {html_file_path}")
            return False
        
        logger.info(f"Converting {html_file_path} to PDF...")
        
        # Create font configuration for better typography
        font_config = FontConfiguration()
        
        # Additional CSS to ensure links work properly in PDF
        additional_css = CSS(string="""
        /* Ensure links are preserved in PDF */
        a {
            color: #3498db !important;
            text-decoration: none !important;
        }
        
        a:hover {
            text-decoration: underline !important;
        }
        
        /* Ensure page numbers are visible */
        @page {
            @bottom-center {
                content: "Page " counter(page) !important;
                font-size: 10pt;
                color: #666;
            }
        }
        
        /* Improve chapter spacing for better page breaks */
        .chapter {
            page-break-before: always;
        }
        
        .chapter:first-of-type {
            page-break-before: avoid;
        }
        
        /* Ensure TOC page numbers align properly */
        .page-number {
            font-weight: normal !important;
        }
        
        /* Better handling of code blocks across pages */
        pre {
            page-break-inside: avoid;
        }
        
        /* Ensure headings don't break badly */
        h1, h2, h3 {
            page-break-after: avoid;
        }
        """, font_config=font_config)
        
        # Create HTML object with base URL for resolving relative links
        html_doc = HTML(filename=html_file_path)
        
        # Render to PDF with optimized settings for links
        html_doc.write_pdf(
            output_pdf_path,
            stylesheets=[additional_css],
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
    """Main function to convert the complete book HTML to PDF."""
    
    # Set up paths
    current_dir = Path(__file__).parent
    html_file = current_dir.parent / "output" / "temp" / "html" / "complete_book.html"
    output_dir = current_dir.parent / "output"
    pdf_file = output_dir / "complete_book.pdf"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert HTML to PDF
    success = convert_html_to_pdf(str(html_file), str(pdf_file))
    
    if success:
        print(f"\n✅ Success! PDF created at: {pdf_file}")
        print(f"📄 File size: {pdf_file.stat().st_size / 1024 / 1024:.2f} MB")
        print("\n📋 Features preserved:")
        print("  • Internal hyperlinks (href links)")
        print("  • Page numbers at bottom of each page")
        print("  • Table of contents with clickable links")
        print("  • Chapter navigation")
        print("  • Code syntax highlighting")
        print("  • Proper page breaks")
    else:
        print("❌ Failed to create PDF. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()

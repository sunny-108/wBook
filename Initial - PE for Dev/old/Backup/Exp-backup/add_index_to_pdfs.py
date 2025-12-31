#!/usr/bin/env python
"""
Script to create an index PDF and append it to all existing PDF books in the output directory.
"""

import os
import sys
from pathlib import Path
import logging
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('add_index_to_pdfs')

try:
    from weasyprint import HTML, CSS
    from PyPDF2 import PdfMerger
except ImportError:
    logger.error("Required packages not installed. Please run: pip install weasyprint PyPDF2")
    sys.exit(1)

# Directory paths
WORKSPACE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = WORKSPACE_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"
TEMP_HTML_DIR = TEMP_DIR / "html"

# Ensure directories exist
TEMP_DIR.mkdir(exist_ok=True)
TEMP_HTML_DIR.mkdir(exist_ok=True)

def create_index_pdf():
    """Create a PDF of the index from the HTML template."""
    index_html_path = TEMP_HTML_DIR / "index.html"
    index_pdf_path = TEMP_DIR / "index.pdf"
    
    # Use WeasyPrint to convert HTML to PDF
    HTML(filename=str(index_html_path)).write_pdf(index_pdf_path)
    
    logger.info(f"Created index PDF: {index_pdf_path}")
    return index_pdf_path

def append_index_to_pdfs():
    """Add index page to all PDFs in the output directory."""
    # Create the index PDF
    index_pdf_path = create_index_pdf()
    
    # Find all PDFs in the output directory
    pdf_files = [f for f in OUTPUT_DIR.glob("*.pdf")]
    
    for pdf_file in pdf_files:
        # Create a backup of the original file
        backup_file = OUTPUT_DIR / f"{pdf_file.stem}_backup{pdf_file.suffix}"
        shutil.copy2(pdf_file, backup_file)
        logger.info(f"Created backup of {pdf_file} as {backup_file}")
        
        # Create a PDF merger object
        merger = PdfMerger()
        
        # Add the original PDF
        merger.append(str(pdf_file))
        
        # Add the index page
        merger.append(str(index_pdf_path))
        
        # Write back to the original file
        merger.write(str(pdf_file))
        merger.close()
        
        logger.info(f"Added index to {pdf_file}")

if __name__ == "__main__":
    try:
        append_index_to_pdfs()
        print("Successfully added index to all PDF books in the output directory.")
    except Exception as e:
        logger.exception(f"Error adding index: {e}")
        print(f"Error: {e}")
        sys.exit(1)

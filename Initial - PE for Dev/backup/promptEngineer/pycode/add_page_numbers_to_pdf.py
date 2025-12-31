#!/usr/bin/env python
"""
Script to add page numbers to an existing PDF book, showing page numbers only for chapters.
This script reads the existing book.pdf and creates a new version with page numbers.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('add_page_numbers')

try:
    import PyPDF2
    from PyPDF2 import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import io
except ImportError:
    logger.error("Required packages not installed. Please run: pip install PyPDF2 reportlab")
    sys.exit(1)

# Directory paths
WORKSPACE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
OUTPUT_DIR = WORKSPACE_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"

# Ensure directories exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def extract_chapter_info() -> List[Dict]:
    """Extract chapter information from chapterFlow.md to identify chapter pages."""
    chapter_flow_path = WORKSPACE_DIR / "chapterFlow.md"
    chapters = []
    
    try:
        with open(chapter_flow_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract chapter titles using regex
        chapter_pattern = r'### (\d+)\.\s+(.+?)(?=\n|$)'
        matches = re.findall(chapter_pattern, content, re.MULTILINE)
        
        for chapter_num, title in matches:
            chapters.append({
                'number': int(chapter_num),
                'title': title.strip(),
                'full_title': f"{chapter_num}. {title.strip()}"
            })
        
        logger.info(f"Found {len(chapters)} chapters")
        return chapters
        
    except Exception as e:
        logger.error(f"Error reading chapter flow: {e}")
        return []

def find_chapter_pages(pdf_path: Path, chapters: List[Dict]) -> Dict[int, int]:
    """
    Analyze the PDF to find which pages contain chapter starts.
    Returns a mapping of page numbers to chapter numbers.
    """
    chapter_pages = {}
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    text = page.extract_text()
                    
                    # Look for chapter headings in the text
                    for chapter in chapters:
                        # Try different patterns to match chapter titles
                        patterns = [
                            f"Chapter {chapter['number']}",
                            f"{chapter['number']}. {chapter['title'][:30]}",  # First 30 chars
                            chapter['title'][:50],  # First 50 chars of title
                            f"Introduction to" if chapter['number'] == 1 else None,
                            f"Understanding LLMs" if chapter['number'] == 2 else None,
                            f"Art and Science" if chapter['number'] == 3 else None,
                            f"Essential Prompting" if chapter['number'] == 4 else None,
                            f"Advanced Prompting" if chapter['number'] == 5 else None,
                            f"Building Effective" if chapter['number'] == 6 else None,
                            f"Hands-on Project 1" if chapter['number'] == 7 else None,
                            f"Hands-on Project 2" if chapter['number'] == 8 else None,
                            f"Hands-on Project 3" if chapter['number'] == 9 else None,
                        ]
                        
                        for pattern in patterns:
                            if pattern and pattern.lower() in text.lower():
                                if chapter['number'] not in chapter_pages.values():
                                    chapter_pages[page_num] = chapter['number']
                                    logger.info(f"Found Chapter {chapter['number']} on page {page_num}")
                                    break
                        
                        if page_num in chapter_pages:
                            break
                            
                except Exception as e:
                    logger.warning(f"Error processing page {page_num}: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Error analyzing PDF: {e}")
    
    return chapter_pages

def create_page_number_overlay(page_num: int, chapter_num: int, page_size) -> bytes:
    """Create a PDF overlay with page number for a chapter page."""
    buffer = io.BytesIO()
    
    # Create a canvas
    c = canvas.Canvas(buffer, pagesize=page_size)
    
    # Get page dimensions
    width, height = page_size
    
    # Add page number at the bottom center
    c.setFont("Helvetica", 10)
    page_text = f"Page {page_num} - Chapter {chapter_num}"
    text_width = c.stringWidth(page_text, "Helvetica", 10)
    x = (width - text_width) / 2
    y = 30  # 30 points from bottom
    
    c.drawString(x, y, page_text)
    c.save()
    
    buffer.seek(0)
    return buffer.getvalue()

def add_page_numbers_to_pdf(input_pdf_path: Path, output_pdf_path: Path, chapter_pages: Dict[int, int]):
    """Add page numbers to the PDF for chapter pages only."""
    try:
        with open(input_pdf_path, 'rb') as input_file:
            pdf_reader = PdfReader(input_file)
            pdf_writer = PdfWriter()
            
            total_pages = len(pdf_reader.pages)
            logger.info(f"Processing {total_pages} pages")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                # Check if this page starts a chapter
                if page_num in chapter_pages:
                    chapter_num = chapter_pages[page_num]
                    
                    # Get the page size
                    page_size = (float(page.mediabox.width), float(page.mediabox.height))
                    
                    # Create overlay with page number
                    overlay_pdf_bytes = create_page_number_overlay(page_num, chapter_num, page_size)
                    
                    # Create overlay PDF
                    overlay_buffer = io.BytesIO(overlay_pdf_bytes)
                    overlay_reader = PdfReader(overlay_buffer)
                    overlay_page = overlay_reader.pages[0]
                    
                    # Merge the overlay with the original page
                    page.merge_page(overlay_page)
                
                # Add the page (modified or original) to the writer
                pdf_writer.add_page(page)
            
            # Write the output PDF
            with open(output_pdf_path, 'wb') as output_file:
                pdf_writer.write(output_file)
                
            logger.info(f"Created PDF with page numbers: {output_pdf_path}")
            
    except Exception as e:
        logger.error(f"Error adding page numbers: {e}")
        raise

def create_enhanced_table_of_contents():
    """Create an enhanced table of contents with page numbers."""
    chapters = extract_chapter_info()
    
    # Read the existing book PDF to find chapter pages
    input_pdf = OUTPUT_DIR / "book.pdf"
    if not input_pdf.exists():
        logger.error(f"Input PDF not found: {input_pdf}")
        return None
    
    chapter_pages = find_chapter_pages(input_pdf, chapters)
    
    # Create TOC HTML
    toc_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Table of Contents</title>
    <style>
        @page {
            size: letter;
            margin: 2cm;
        }
        body {
            font-family: "Helvetica", "Arial", sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            font-size: 11pt;
        }
        .toc {
            max-width: 800px;
            margin: 0 auto;
        }
        .toc h1 {
            text-align: center;
            color: #2c3e50;
            font-size: 24pt;
            margin-bottom: 2em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5em;
        }
        .toc-entry {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin: 1em 0;
            border-bottom: 1px dotted #ccc;
            padding-bottom: 0.5em;
        }
        .chapter-title {
            flex: 1;
            font-weight: bold;
            color: #2c3e50;
        }
        .dots {
            flex: 0 1 auto;
            border-bottom: 1px dotted #999;
            margin: 0 10px;
            height: 1px;
            align-self: flex-end;
            margin-bottom: 6px;
        }
        .page-number {
            font-weight: bold;
            color: #3498db;
            min-width: 40px;
            text-align: right;
        }
    </style>
</head>
<body>
    <div class="toc">
        <h1>Table of Contents</h1>
"""
    
    # Add entries for chapters with page numbers
    for chapter in chapters:
        page_num = None
        for page, ch_num in chapter_pages.items():
            if ch_num == chapter['number']:
                page_num = page
                break
        
        if page_num:
            toc_html += f"""
        <div class="toc-entry">
            <span class="chapter-title">{chapter['full_title']}</span>
            <span class="dots"></span>
            <span class="page-number">{page_num}</span>
        </div>"""
        else:
            toc_html += f"""
        <div class="toc-entry">
            <span class="chapter-title">{chapter['full_title']}</span>
            <span class="dots"></span>
            <span class="page-number">--</span>
        </div>"""
    
    toc_html += """
    </div>
</body>
</html>"""
    
    # Save the TOC HTML
    toc_html_path = TEMP_DIR / "toc_with_pages.html"
    with open(toc_html_path, 'w', encoding='utf-8') as f:
        f.write(toc_html)
    
    logger.info(f"Created TOC with page numbers: {toc_html_path}")
    return chapter_pages

def main():
    """Main function to add page numbers to the PDF."""
    print("Adding page numbers to book.pdf...")
    
    # Extract chapter information and find chapter pages
    chapter_pages = create_enhanced_table_of_contents()
    
    if not chapter_pages:
        logger.error("Could not determine chapter page locations")
        return
    
    # Input and output paths
    input_pdf = OUTPUT_DIR / "book.pdf"
    output_pdf = OUTPUT_DIR / "book_with_page_numbers.pdf"
    
    if not input_pdf.exists():
        logger.error(f"Input PDF not found: {input_pdf}")
        return
    
    # Add page numbers to the PDF
    try:
        add_page_numbers_to_pdf(input_pdf, output_pdf, chapter_pages)
        print(f"\n✅ Success! Created: {output_pdf}")
        print(f"📄 Added page numbers to {len(chapter_pages)} chapter pages")
        
        # Show which chapters were found
        print("\n📋 Chapter page locations:")
        for page_num, chapter_num in sorted(chapter_pages.items()):
            print(f"   Chapter {chapter_num}: Page {page_num}")
            
    except Exception as e:
        logger.error(f"Failed to add page numbers: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

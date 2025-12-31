#!/usr/bin/env python
"""
Script to add page numbers to the book by regenerating it with page number information.
This script analyzes the existing PDF, determines chapter page locations, and creates 
a new version with page numbers displayed only for chapter start pages.
"""

import os
import sys
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('add_chapter_page_numbers')

try:
    from weasyprint import HTML, CSS
    import PyPDF2
    from PyPDF2 import PdfReader
    import markdown
except ImportError:
    logger.error("Required packages not installed. Please run: pip install weasyprint PyPDF2 markdown")
    sys.exit(1)

# Directory paths
WORKSPACE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
OUTPUT_DIR = WORKSPACE_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"
TEMP_HTML_DIR = TEMP_DIR / "html"

# Ensure directories exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)
TEMP_HTML_DIR.mkdir(parents=True, exist_ok=True)

def extract_chapter_info() -> List[Dict]:
    """Extract chapter information from chapterFlow.md."""
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

def analyze_existing_pdf_for_pages() -> Dict[int, int]:
    """
    Analyze the existing PDF to determine approximate chapter page locations.
    This is a heuristic approach based on content analysis.
    """
    pdf_path = OUTPUT_DIR / "book.pdf"
    chapter_pages = {}
    total_pages = 0
    
    if not pdf_path.exists():
        logger.warning("book.pdf not found, will estimate page numbers")
        return {}
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            total_pages = len(pdf_reader.pages)
            logger.info(f"Analyzing {total_pages} pages in existing PDF")
            
            # Heuristic: Assume chapters are roughly evenly distributed
            # and look for chapter-like content
            estimated_pages_per_chapter = max(1, total_pages // 9)  # 9 chapters
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    text = page.extract_text().lower()
                    
                    # Look for chapter indicators
                    chapter_indicators = [
                        ("introduction", 1),
                        ("understanding llm", 2), 
                        ("art and science", 3),
                        ("essential prompting", 4),
                        ("advanced prompting", 5),
                        ("building effective", 6),
                        ("hands-on project 1", 7),
                        ("hands-on project 2", 8),
                        ("hands-on project 3", 9),
                    ]
                    
                    for indicator, chapter_num in chapter_indicators:
                        if indicator in text and chapter_num not in chapter_pages.values():
                            chapter_pages[page_num] = chapter_num
                            logger.info(f"Found Chapter {chapter_num} on page {page_num}")
                            break
                            
                except Exception as e:
                    logger.warning(f"Error processing page {page_num}: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Error analyzing PDF: {e}")
    
    # If we couldn't find all chapters, estimate based on content distribution
    if len(chapter_pages) < 9:
        logger.info("Estimating remaining chapter locations...")
        estimated_pages_per_chapter = max(1, total_pages // 9) if total_pages > 0 else 15
        estimated_pages = {}
        for i in range(1, 10):
            if i not in chapter_pages.values():
                estimated_page = max(1, i * estimated_pages_per_chapter)
                estimated_pages[estimated_page] = i
        
        # Merge estimated pages with found pages
        chapter_pages.update(estimated_pages)
    
    return chapter_pages

def get_chapter_files() -> List[Path]:
    """Get all chapter markdown files in order."""
    chapter_files = []
    
    # Look for chapter files
    for i in range(1, 10):
        chapter_file = WORKSPACE_DIR / f"chapter{i}.md"
        if chapter_file.exists():
            chapter_files.append(chapter_file)
        else:
            logger.warning(f"Chapter file not found: {chapter_file}")
    
    logger.info(f"Found {len(chapter_files)} chapter files")
    return chapter_files

def create_html_with_page_numbers():
    """Create a complete HTML file with page number indicators for chapters."""
    
    # Get chapter information and page locations
    chapters = extract_chapter_info()
    chapter_pages = analyze_existing_pdf_for_pages()
    chapter_files = get_chapter_files()
    
    # CSS with page numbers for chapters
    css_style = """
@page {
    size: letter;
    margin: 2cm;
    @bottom-center {
        content: "Page " counter(page);
    }
}

body {
    font-family: "Helvetica", "Arial", sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 0 0.5cm;
    font-size: 10.5pt;
    counter-reset: chapter;
}

/* Chapter page styling with visible page numbers */
.chapter-container {
    page-break-before: always !important;
    break-before: page !important;
    padding-top: 60px;
    min-height: 100vh;
    position: relative;
    margin-bottom: 40px;
}

.chapter-container:first-of-type {
    page-break-before: avoid !important;
    break-before: avoid !important;
}

/* Chapter header with page number display */
.chapter-header {
    position: relative;
    margin-bottom: 2em;
    padding-bottom: 1em;
    border-bottom: 2px solid #3498db;
}

.chapter-page-number {
    position: absolute;
    top: -40px;
    right: 0;
    background: #3498db;
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 12pt;
    font-weight: bold;
}

h1 {
    font-size: 20pt;
    color: #2c3e50;
    margin-top: 0;
    margin-bottom: 0.5em;
}

h2 {
    font-size: 16pt;
    color: #3498db;
    margin-top: 1em;
    page-break-after: avoid;
}

h3 {
    font-size: 14pt;
    color: #2980b9;
    page-break-after: avoid;
}

/* Table of Contents styling */
.toc {
    page-break-after: always;
    max-width: 800px;
    margin: 0 auto 2em auto;
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

/* Code styling */
pre {
    background-color: #f8f8f8;
    border: 1px solid #ddd;
    border-radius: 3px;
    padding: 1em;
    overflow-x: auto;
    page-break-inside: avoid;
}

code {
    font-family: "Courier New", monospace;
    background-color: #f8f8f8;
    padding: 2px 4px;
    border-radius: 3px;
}

/* Table styling */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    page-break-inside: avoid;
}

table, th, td {
    border: 1px solid #ddd;
}

th, td {
    padding: 0.5em;
    text-align: left;
}

th {
    background-color: #f2f2f2;
}

blockquote {
    border-left: 4px solid #ccc;
    padding-left: 1em;
    color: #555;
    margin: 1em 0;
}

img {
    max-width: 100%;
    page-break-inside: avoid;
}
"""

    # Start building the HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Complete Book with Page Numbers</title>
    <style>
{css_style}
    </style>
</head>
<body>
"""

    # Add Table of Contents with page numbers
    html_content += '<div class="toc">\n<h1>Table of Contents</h1>\n'
    
    for chapter in chapters:
        page_num = None
        for page, ch_num in chapter_pages.items():
            if ch_num == chapter['number']:
                page_num = page
                break
        
        if page_num:
            html_content += f'''
<div class="toc-entry">
    <span class="chapter-title">{chapter['full_title']}</span>
    <span class="dots"></span>
    <span class="page-number">{page_num}</span>
</div>'''
        else:
            # Estimate page number if not found
            estimated_page = (chapter['number'] - 1) * 15 + 3  # Rough estimate
            html_content += f'''
<div class="toc-entry">
    <span class="chapter-title">{chapter['full_title']}</span>
    <span class="dots"></span>
    <span class="page-number">~{estimated_page}</span>
</div>'''
    
    html_content += '</div>\n'
    
    # Add chapters with page number indicators
    for i, chapter_file in enumerate(chapter_files, 1):
        try:
            with open(chapter_file, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Convert markdown to HTML
            html_chapter = markdown.markdown(md_content, extensions=['tables', 'codehilite'])
            
            # Find the page number for this chapter
            page_num = None
            for page, ch_num in chapter_pages.items():
                if ch_num == i:
                    page_num = page
                    break
            
            if not page_num:
                page_num = (i - 1) * 15 + 3  # Estimate
            
            # Wrap chapter in container with page number
            html_content += f'''
<div class="chapter-container">
    <div class="chapter-header">
        <div class="chapter-page-number">Page {page_num}</div>
    </div>
    {html_chapter}
</div>
'''
            
        except Exception as e:
            logger.error(f"Error processing {chapter_file}: {e}")
            continue
    
    html_content += '</body>\n</html>'
    
    # Save the HTML file
    html_output_path = TEMP_HTML_DIR / "book_with_page_numbers.html"
    with open(html_output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Created HTML with page numbers: {html_output_path}")
    return html_output_path

def convert_html_to_pdf(html_path: Path) -> Path:
    """Convert the HTML file to PDF using WeasyPrint."""
    pdf_output_path = OUTPUT_DIR / "book_with_page_numbers.pdf"
    
    try:
        # Convert HTML to PDF
        HTML(filename=str(html_path)).write_pdf(str(pdf_output_path))
        logger.info(f"Created PDF with page numbers: {pdf_output_path}")
        return pdf_output_path
        
    except Exception as e:
        logger.error(f"Error converting HTML to PDF: {e}")
        raise

def main():
    """Main function to create book with page numbers."""
    print("Creating book with chapter page numbers...")
    
    try:
        # Create HTML with page numbers
        html_path = create_html_with_page_numbers()
        
        # Convert to PDF
        pdf_path = convert_html_to_pdf(html_path)
        
        print(f"\n✅ Success! Created: {pdf_path}")
        print("📄 Page numbers are displayed for each chapter")
        print("📋 Table of Contents includes page numbers")
        
        # Show chapter information
        chapters = extract_chapter_info()
        chapter_pages = analyze_existing_pdf_for_pages()
        
        print(f"\n📖 Book contains {len(chapters)} chapters:")
        for chapter in chapters:
            page_num = None
            for page, ch_num in chapter_pages.items():
                if ch_num == chapter['number']:
                    page_num = page
                    break
            if page_num:
                print(f"   Chapter {chapter['number']}: Page {page_num}")
            else:
                estimated = (chapter['number'] - 1) * 15 + 3
                print(f"   Chapter {chapter['number']}: Page ~{estimated} (estimated)")
        
    except Exception as e:
        logger.error(f"Failed to create book with page numbers: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

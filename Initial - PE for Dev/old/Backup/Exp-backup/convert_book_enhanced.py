#!/usr/bin/env python
"""
Enhanced script to convert Markdown chapter files to PDF with proper table of contents,
clickable navigation and improved formatting.
"""

import os
import re
import sys
import tempfile
import shutil
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import subprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('md2pdf_enhanced')

try:
    import markdown
    import PyPDF2
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
except ImportError:
    logger.error("Required packages not installed. Please run: pip install markdown PyPDF2 weasyprint")
    sys.exit(1)

# Directory where script is running
WORKSPACE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Configure output directories
OUTPUT_DIR = WORKSPACE_DIR / "output"
TEMP_DIR = OUTPUT_DIR / "temp"

# CSS for PDF styling with improved formatting
CSS_STYLE = """
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
}

h1 {
    font-size: 20pt;
    color: #2c3e50;
    page-break-before: always;
    margin-top: 1.5cm;
}

h1:first-of-type {
    page-break-before: avoid;
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
    page-break-before: always;
    page-break-after: always;
}

.toc h1 {
    text-align: center;
    font-size: 24pt;
    margin-bottom: 1.5em;
}

.toc ul {
    list-style-type: none;
    padding-left: 0;
}

.toc ul ul {
    padding-left: 1.5em;
}

.toc li {
    margin-bottom: 0.5em;
    page-break-inside: avoid;
}

.toc-entry {
    display: flex;
    align-items: baseline;
    width: 100%;
}

.chapter-number {
    display: inline-block;
    width: 1.5em;
    font-weight: bold;
}

.chapter-title {
    font-weight: bold;
}

.dots {
    flex: 1;
    margin: 0 0.5em;
    border-bottom: 1px dotted #999;
    min-width: 2em;
}

.page-number {
    text-align: right;
    min-width: 2em;
}

.section-entry {
    font-size: 95%;
    margin-top: 0.3em;
    margin-bottom: 0.3em;
}

/* Improved code formatting */
pre {
    background-color: #f8f8f8;
    border: 1px solid #ddd;
    border-radius: 3px;
    padding: 1em;
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    font-size: 9pt;
    line-height: 1.4;
    font-family: "Courier New", monospace;
    max-width: 100%;
    overflow-x: hidden;
}

code {
    font-family: "Courier New", monospace;
    background-color: #f8f8f8;
    padding: 2px 4px;
    border-radius: 3px;
    font-size: 9pt;
    word-wrap: break-word;
    white-space: pre-wrap;
}

blockquote {
    border-left: 4px solid #ccc;
    padding-left: 1em;
    color: #555;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 9.5pt;
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

img {
    max-width: 100%;
}
"""

def ensure_directories_exist():
    """Create output and temp directories if they don't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    logger.info(f"Created output directories: {OUTPUT_DIR} and {TEMP_DIR}")

def get_chapter_files() -> List[Path]:
    """Get all chapter markdown files sorted numerically."""
    # Get all chapter files
    all_files = [f for f in WORKSPACE_DIR.glob("chapter*.md")]
    
    # Create list of (file, chapter_number) tuples
    file_with_numbers = []
    for f in all_files:
        match = re.search(r'chapter(\d+)\.md', f.name)
        if match:
            chapter_num = int(match.group(1))
            file_with_numbers.append((f, chapter_num))
        else:
            # For files like "chapter.md" without numbers, assign a high number
            file_with_numbers.append((f, 1000))
    
    # Sort by chapter number
    file_with_numbers.sort(key=lambda x: x[1])
    
    # Return just the files in order
    chapter_files = [f[0] for f in file_with_numbers]
    
    logger.info(f"Found {len(chapter_files)} chapter files")
    return chapter_files

def parse_toc_from_chapter_flow() -> List[Dict]:
    """Parse the table of contents structure from chapterFlow.md."""
    toc_path = WORKSPACE_DIR / "chapterFlow.md"
    if not toc_path.exists():
        logger.warning(f"TOC file {toc_path} not found. Will not generate outline.")
        return []
        
    with open(toc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract chapters and their titles
    toc_entries = []
    
    # Match chapter headings (like "### 1. Title")
    chapter_pattern = r'### (\d+)\. (.*)'
    current_chapter = None
    
    for line in content.split('\n'):
        chapter_match = re.match(chapter_pattern, line)
        if chapter_match:
            chapter_num = int(chapter_match.group(1))
            chapter_title = chapter_match.group(2).strip()
            current_chapter = {
                'number': chapter_num,
                'title': chapter_title,
                'sections': []
            }
            toc_entries.append(current_chapter)
        elif line.startswith('- ') and current_chapter:
            # Get section titles (bullet points under chapter headings)
            section_title = line[2:].strip()
            current_chapter['sections'].append(section_title)
    
    logger.info(f"Parsed TOC with {len(toc_entries)} chapters")
    return toc_entries

def convert_md_to_pdf(md_file: Path) -> Path:
    """Convert a markdown file to PDF using WeasyPrint."""
    pdf_output = TEMP_DIR / f"{md_file.stem}.pdf"
    
    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=[
            'markdown.extensions.extra',
            'markdown.extensions.codehilite',
            'markdown.extensions.toc'
        ]
    )
    
    # Add ID to chapter heading for PDF linking
    chapter_match = re.search(r'chapter(\d+)', md_file.stem)
    if chapter_match:
        chapter_num = chapter_match.group(1)
        # Add id attribute to the first h1 heading
        html_content = re.sub(
            r'<h1>(.+?)</h1>',
            f'<h1 id="chapter{chapter_num}">\\1</h1>',
            html_content,
            count=1
        )
    
    # Wrap HTML with proper structure
    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{md_file.stem}</title>
        <style>{CSS_STYLE}</style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """
    
    # Convert HTML to PDF using WeasyPrint
    font_config = FontConfiguration()
    HTML(string=html_doc).write_pdf(
        pdf_output,
        stylesheets=[CSS(string=CSS_STYLE)],
        font_config=font_config
    )
    
    logger.info(f"Converted {md_file} to {pdf_output}")
    return pdf_output

def get_pdf_page_count(pdf_path: Path) -> int:
    """Get the number of pages in a PDF file."""
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        return len(pdf_reader.pages)

def calculate_page_numbers(pdfs: List[Path]) -> Dict[str, int]:
    """Calculate starting page numbers for each chapter."""
    page_numbers = {}
    current_page = 1
    
    for pdf_path in pdfs:
        chapter_match = re.search(r'chapter(\d+)', pdf_path.stem)
        if chapter_match:
            chapter_num = chapter_match.group(1)
            page_numbers[f"chapter{chapter_num}"] = current_page
        
        # Update current page for next PDF
        page_count = get_pdf_page_count(pdf_path)
        current_page += page_count
    
    return page_numbers

def create_toc_pdf(page_numbers: Dict[str, int]) -> Path:
    """Create a PDF with the table of contents."""
    toc_entries = parse_toc_from_chapter_flow()
    
    # Generate TOC HTML
    toc_html = """
    <div class="toc">
        <h1>Contents</h1>
        <ul>
    """
    
    for entry in toc_entries:
        chapter_num = entry['number']
        chapter_title = entry['title']
        chapter_id = f"chapter{chapter_num}"
        
        page_num = page_numbers.get(chapter_id, "#")
        
        toc_html += f"""
        <li>
            <div class="toc-entry">
                <span>
                    <span class="chapter-number">{chapter_num}.</span>
                    <span class="chapter-title">{chapter_title}</span>
                </span>
                <span class="dots"></span>
                <span class="page-number">{page_num}</span>
            </div>
            <ul>
        """
        
        for section in entry['sections']:
            # We're using the same page number for sections as chapters
            # In a more advanced implementation, you could calculate more precise section page numbers
            toc_html += f"""
            <li class="section-entry">
                <div class="toc-entry">
                    <span>{section}</span>
                    <span class="dots"></span>
                    <span class="page-number">{page_num}</span>
                </div>
            </li>
            """
        
        toc_html += """
            </ul>
        </li>
        """
    
    toc_html += """
        </ul>
    </div>
    """
    
    # Wrap HTML with proper structure
    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Table of Contents</title>
        <style>{CSS_STYLE}</style>
    </head>
    <body>
    {toc_html}
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    toc_pdf_path = TEMP_DIR / "contents.pdf"
    font_config = FontConfiguration()
    HTML(string=html_doc).write_pdf(
        toc_pdf_path,
        stylesheets=[CSS(string=CSS_STYLE)],
        font_config=font_config
    )
    
    logger.info(f"Created table of contents PDF: {toc_pdf_path}")
    return toc_pdf_path

def create_bookmarks(toc_entries: List[Dict], page_numbers: Dict[str, int]) -> List[Dict]:
    """Create PDF bookmarks structure based on TOC entries and page numbers."""
    bookmarks = []
    
    # Create bookmarks with correct page numbers
    for entry in toc_entries:
        chapter_num = entry['number']
        chapter_key = f"chapter{chapter_num}"
        
        if chapter_key in page_numbers:
            # Add chapter bookmark
            chapter_page = page_numbers[chapter_key]
            chapter_bookmark = {
                'title': f"Chapter {chapter_num}: {entry['title']}",
                'page': chapter_page,
                'children': []
            }
            bookmarks.append(chapter_bookmark)
            
            # Add section bookmarks - for simplicity, we use the same page as the chapter
            # In a more advanced implementation, you could calculate more precise section page numbers
            for i, section in enumerate(entry['sections']):
                section_bookmark = {
                    'title': section,
                    'page': chapter_page,
                    'children': []
                }
                chapter_bookmark['children'].append(section_bookmark)
    
    return bookmarks

def merge_pdfs(pdf_files: List[Path], output_path: Path, bookmarks: List[Dict]):
    """Merge multiple PDFs into one with bookmarks."""
    merger = PyPDF2.PdfMerger()
    
    # Add each PDF file to the merger
    for pdf_file in pdf_files:
        merger.append(str(pdf_file))
    
    # Add bookmarks
    for bookmark in bookmarks:
        add_bookmarks_recursively(merger, bookmark, None)
    
    # Write the merged PDF
    merger.write(str(output_path))
    merger.close()
    
    logger.info(f"Merged {len(pdf_files)} PDFs into {output_path}")

def add_bookmarks_recursively(merger, bookmark, parent):
    """Add bookmarks recursively to the PDF."""
    new_bookmark = merger.add_outline_item(
        bookmark['title'], bookmark['page'], parent=parent
    )
    
    for child in bookmark.get('children', []):
        add_bookmarks_recursively(merger, child, new_bookmark)

def add_front_matter_pdfs(about_book_path: Path, about_author_path: Path) -> List[Path]:
    """Convert front matter markdown files to PDFs if they exist."""
    front_matter_pdfs = []
    
    # Convert about_book.md if it exists
    if about_book_path.exists():
        front_matter_pdfs.append(convert_md_to_pdf(about_book_path))
    
    # Convert about_author.md if it exists
    if about_author_path.exists():
        front_matter_pdfs.append(convert_md_to_pdf(about_author_path))
    
    return front_matter_pdfs

def clean_up_temp_files():
    """Remove temporary files."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        logger.info(f"Removed temporary directory: {TEMP_DIR}")

def main():
    """Main function to convert markdown chapters to PDF and merge them."""
    try:
        # Ensure output directories exist
        ensure_directories_exist()
        
        # Get all chapter files
        chapter_files = get_chapter_files()
        if not chapter_files:
            logger.error("No chapter files found!")
            return
        
        # Parse TOC from chapterFlow.md
        toc_entries = parse_toc_from_chapter_flow()
        
        # Convert front matter files to PDF
        front_matter_pdfs = add_front_matter_pdfs(
            WORKSPACE_DIR / "about_book.md",
            WORKSPACE_DIR / "about_author.md"
        )
        
        # Convert each chapter to PDF
        chapter_pdf_files = []
        for md_file in chapter_files:
            pdf_file = convert_md_to_pdf(md_file)
            chapter_pdf_files.append(pdf_file)
        
        # Calculate page numbers for TOC
        all_pdfs = front_matter_pdfs + chapter_pdf_files
        page_numbers = calculate_page_numbers(all_pdfs)
        
        # Create table of contents PDF
        toc_pdf = create_toc_pdf(page_numbers)
        
        # Create bookmarks with accurate page numbers
        bookmarks = create_bookmarks(toc_entries, page_numbers)
        
        # Merge all PDFs (front matter + TOC + chapters)
        all_pdfs_with_toc = front_matter_pdfs + [toc_pdf] + chapter_pdf_files
        output_path = OUTPUT_DIR / "book_enhanced.pdf"
        merge_pdfs(all_pdfs_with_toc, output_path, bookmarks)
        
        logger.info(f"Book successfully created with interactive table of contents: {output_path}")
        
        # Clean up temporary files AFTER we're done using them
        clean_up_temp_files()
        
    except Exception as e:
        logger.exception(f"Error during conversion process: {e}")
        raise

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Script to convert Markdown chapter files to PDF with proper page breaks, navigation and TOC.
This version ensures every chapter starts on a new page.
"""

import os
import re
import sys
import tempfile
import shutil
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import subprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('md2pdf_page_breaks')

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
TEMP_HTML_DIR = TEMP_DIR / "html"

# CSS for PDF styling with improved formatting and link styling
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

/* Strong chapter break styles */
.chapter-container {
    page-break-before: always !important;
    break-before: page !important;
    padding-top: 60px; /* Add space at the top of each chapter */
    margin-bottom: 40px; /* Add space at bottom of chapter */
}

/* First chapter doesn't need a page break */
.chapter-container:first-of-type {
    page-break-before: avoid !important;
    break-before: avoid !important;
}

h1 {
    font-size: 20pt;
    color: #2c3e50;
    margin-top: 1.5cm;
    page-break-after: avoid;
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

.toc a {
    text-decoration: none;
    color: #0066cc;
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

/* Link styling */
a {
    color: #0066cc;
    text-decoration: none;
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
    TEMP_HTML_DIR.mkdir(exist_ok=True)
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

def generate_combined_html(chapter_files: List[Path], toc_entries: List[Dict]) -> str:
    """Generate combined HTML from all chapters."""
    chapter_contents = []
    chapter_map = {}  # Map chapter numbers to their HTML content
    section_map = {}  # Store sections for each chapter
    
    # First pass: Read markdown content and get section titles
    for chapter_file in chapter_files:
        chapter_num = None
        match = re.search(r'chapter(\d+)', chapter_file.stem)
        if match:
            chapter_num = int(match.group(1))
            
            # Read markdown content
            with open(chapter_file, 'r', encoding='utf-8') as f:
                md_content = f.read()
                
            # Extract section titles (assuming they're h2 headers in markdown)
            section_titles = []
            for line in md_content.split('\n'):
                if line.startswith('## '):
                    section_titles.append(line[3:].strip())
            
            section_map[chapter_num] = section_titles
    
    # Second pass: Generate HTML with proper section IDs
    for chapter_file in chapter_files:
        chapter_num = None
        match = re.search(r'chapter(\d+)', chapter_file.stem)
        if match:
            chapter_num = int(match.group(1))
            
        # Read markdown content
        with open(chapter_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
            
        # Convert to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=[
                'markdown.extensions.extra',
                'markdown.extensions.codehilite',
                'markdown.extensions.toc'
            ]
        )
        
        if chapter_num is not None:
            # Add ID to chapter heading (h1)
            html_content = re.sub(
                r'<h1>(.*?)</h1>',
                f'<h1 id="chapter{chapter_num}">\\1</h1>',
                html_content,
                count=1
            )
            
            # Add IDs to section headings (h2) that match expected sections
            for i, section_title in enumerate(section_map.get(chapter_num, [])):
                section_num = i + 1
                section_id = f"chapter{chapter_num}_section{section_num}"
                
                # Escape special regex characters
                escaped_title = re.escape(section_title)
                # Add ID to the h2 with this title
                html_content = re.sub(
                    f'<h2>({escaped_title})</h2>',
                    f'<h2 id="{section_id}">\\1</h2>',
                    html_content
                )
            
            # Wrap the whole chapter in a div with the chapter-container class
            html_content = f'<div class="chapter-container" id="chapter{chapter_num}">\n{html_content}\n</div>'
            
            chapter_map[chapter_num] = html_content
        else:
            # If it's not a numbered chapter, just add the content
            chapter_contents.append(html_content)
    
    # Generate table of contents HTML
    toc_html = create_toc_html(toc_entries, chapter_map)
    
    # Add front matter if it exists
    front_matter = []
    about_book_path = WORKSPACE_DIR / "about_book.md"
    about_author_path = WORKSPACE_DIR / "about_author.md"
    
    if about_book_path.exists():
        with open(about_book_path, 'r', encoding='utf-8') as f:
            about_book_md = f.read()
        about_book_html = markdown.markdown(
            about_book_md,
            extensions=[
                'markdown.extensions.extra',
                'markdown.extensions.codehilite'
            ]
        )
        front_matter.append(about_book_html)
    
    if about_author_path.exists():
        with open(about_author_path, 'r', encoding='utf-8') as f:
            about_author_md = f.read()
        about_author_html = markdown.markdown(
            about_author_md,
            extensions=[
                'markdown.extensions.extra',
                'markdown.extensions.codehilite'
            ]
        )
        front_matter.append(about_author_html)
    
    # Combine all HTML content in the right order
    combined_html = ""
    
    # Add front matter
    for content in front_matter:
        combined_html += content + "\n\n"
    
    # Add table of contents
    combined_html += toc_html + "\n\n"
    
    # Add chapters in order
    for entry in toc_entries:
        chapter_num = entry['number']
        if chapter_num in chapter_map:
            combined_html += chapter_map[chapter_num] + "\n\n"
    
    # Add any remaining chapters not in the TOC
    for content in chapter_contents:
        if content not in combined_html:
            combined_html += content + "\n\n"
    
    return combined_html

def create_toc_html(toc_entries: List[Dict], chapter_map: Dict) -> str:
    """Create HTML for table of contents with links to chapters."""
    toc_html = """
    <div class="toc">
        <h1>Contents</h1>
        <ul>
    """
    
    for entry in toc_entries:
        chapter_num = entry['number']
        chapter_title = entry['title']
        
        if chapter_num in chapter_map:
            # Create link to chapter
            toc_html += f"""
            <li>
                <div class="toc-entry">
                    <a href="#chapter{chapter_num}">
                        <span class="chapter-number">{chapter_num}.</span>
                        <span class="chapter-title">{chapter_title}</span>
                    </a>
                    <span class="dots"></span>
                    <span class="page-number"></span>
                </div>
                <ul>
            """
            
            # Add links to sections
            for i, section in enumerate(entry['sections']):
                section_num = i + 1
                toc_html += f"""
                <li class="section-entry">
                    <div class="toc-entry">
                        <a href="#chapter{chapter_num}_section{section_num}">{section}</a>
                        <span class="dots"></span>
                        <span class="page-number"></span>
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
    
    return toc_html

def create_combined_pdf(combined_html: str, output_path: Path) -> Path:
    """Create a PDF from combined HTML content."""
    # Save combined HTML to file
    html_path = TEMP_HTML_DIR / "combined.html"
    
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Book</title>
    <style>{CSS_STYLE}</style>
</head>
<body>
{combined_html}
</body>
</html>"""
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    # Convert HTML to PDF
    HTML(filename=str(html_path)).write_pdf(
        output_path,
        stylesheets=[CSS(string=CSS_STYLE)]
    )
    
    logger.info(f"Created PDF: {output_path}")
    return output_path

def create_bookmarks(toc_entries: List[Dict], pdf_path: Path) -> None:
    """Add bookmarks to the PDF."""
    # Create temporary file for output PDF
    temp_output = TEMP_DIR / "temp_output.pdf"
    
    # Create PDF reader
    reader = PyPDF2.PdfReader(pdf_path)
    
    # Create PDF writer with updated bookmarks
    writer = PyPDF2.PdfWriter()
    
    # Copy all pages from reader to writer
    for page in reader.pages:
        writer.add_page(page)
    
    # Add bookmarks for chapters and sections
    for entry in toc_entries:
        chapter_num = entry['number']
        chapter_title = entry['title']
        
        # Add chapter bookmark - we're approximating page numbers here
        chapter_page = chapter_num - 1  # Approximate for now
        if chapter_page < 0:
            chapter_page = 0
        
        # Add chapter bookmark
        chapter_bookmark = writer.add_outline_item(
            f"Chapter {chapter_num}: {chapter_title}", 
            chapter_page
        )
        
        # Add section bookmarks as children
        for i, section in enumerate(entry['sections']):
            section_page = chapter_page + i  # Approximate
            writer.add_outline_item(
                section, 
                section_page, 
                parent=chapter_bookmark
            )
    
    # Write output PDF
    with open(temp_output, 'wb') as f:
        writer.write(f)
    
    # Replace original PDF with bookmarked version
    shutil.copy(temp_output, pdf_path)

def main():
    """Main function to convert markdown files to PDF."""
    try:
        # Create output directories
        ensure_directories_exist()
        
        # Get all chapter files
        chapter_files = get_chapter_files()
        
        # Parse TOC from chapter flow
        toc_entries = parse_toc_from_chapter_flow()
        
        # Generate combined HTML from all chapters
        combined_html = generate_combined_html(chapter_files, toc_entries)
        
        # Create output PDF path
        output_path = OUTPUT_DIR / "book_with_page_breaks.pdf"
        
        # Create combined PDF
        pdf_path = create_combined_pdf(combined_html, output_path)
        
        # Add bookmarks to PDF
        create_bookmarks(toc_entries, pdf_path)
        
        logger.info(f"Successfully created book PDF with proper page breaks: {pdf_path}")
        print(f"Success! Book PDF created at: {pdf_path}")
        
    except Exception as e:
        logger.error(f"Error converting book: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

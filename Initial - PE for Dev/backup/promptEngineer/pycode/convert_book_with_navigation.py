#!/usr/bin/env python
"""
Navigation-focused script to convert Markdown chapter files to PDF with properly working links.
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
logger = logging.getLogger('md2pdf_navigation')

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

h1 {
    font-size: 20pt;
    color: #2c3e50;
    page-break-before: always;
    margin-top: 1.5cm;
}

h1:first-of-type {
    page-break-before: avoid;
}

/* Add chapter container for better page break control */
.chapter-container {
    page-break-before: always;
}

.chapter-container:first-of-type {
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

def add_chapter_anchors_to_html(html_content: str, chapter_num: Optional[int] = None) -> str:
    """Add anchor IDs to HTML headings for navigation."""
    if chapter_num is None:
        return html_content
    
    # Add ID to chapter heading (h1) and wrap in a div with class chapter-container
    modified_html = re.sub(
        r'<h1>(.*?)</h1>',
        f'<div class="chapter-container"><h1 id="chapter{chapter_num}">\\1</h1>',
        html_content,
        count=1
    )
    
    # Close the chapter-container div at the end
    modified_html = modified_html + "\n</div>"
    
    # Add IDs to section headings (h2)
    section_num = 1
    pattern = r'<h2>(.*?)</h2>'
    
    def add_section_id(match):
        nonlocal section_num
        section_id = f"chapter{chapter_num}_section{section_num}"
        section_num += 1
        return f'<h2 id="{section_id}">{match.group(1)}</h2>'
    
    modified_html = re.sub(pattern, add_section_id, modified_html)
    
    return modified_html

def convert_md_to_html(md_file: Path) -> Tuple[str, Optional[int]]:
    """Convert markdown file to HTML with proper anchors."""
    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Check if this is a chapter file to add anchors
    chapter_num = None
    chapter_match = re.search(r'chapter(\d+)', md_file.stem)
    if chapter_match:
        chapter_num = int(chapter_match.group(1))
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=[
            'markdown.extensions.extra',
            'markdown.extensions.codehilite',
            'markdown.extensions.toc'
        ]
    )
    
    # Add anchor IDs to headings
    html_content = add_chapter_anchors_to_html(html_content, chapter_num)
    
    return html_content, chapter_num

def save_html_file(html_content: str, title: str, filename: str) -> Path:
    """Save HTML content to a file."""
    html_path = TEMP_HTML_DIR / filename
    
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>{CSS_STYLE}</style>
</head>
<body>
{html_content}
</body>
</html>"""
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(full_html)
    
    return html_path

def html_to_pdf(html_path: Path, pdf_path: Path) -> Path:
    """Convert HTML file to PDF."""
    HTML(filename=str(html_path)).write_pdf(
        pdf_path,
        stylesheets=[CSS(string=CSS_STYLE)]
    )
    
    logger.info(f"Converted {html_path} to {pdf_path}")
    return pdf_path

def convert_chapter_to_pdf(md_file: Path) -> Tuple[Path, Optional[int]]:
    """Convert a chapter markdown file to PDF with proper anchors."""
    # First convert to HTML with anchors
    html_content, chapter_num = convert_md_to_html(md_file)
    
    # Save HTML file
    html_filename = f"{md_file.stem}.html"
    html_path = save_html_file(html_content, md_file.stem, html_filename)
    
    # Convert HTML to PDF
    pdf_path = TEMP_DIR / f"{md_file.stem}.pdf"
    html_to_pdf(html_path, pdf_path)
    
    return pdf_path, chapter_num

def create_toc_html(toc_entries: List[Dict], chapter_files: List[Path]) -> str:
    """Create HTML for table of contents with links to chapters."""
    toc_html = """
    <div class="toc">
        <h1>Contents</h1>
        <ul>
    """
    
    # Create a mapping of chapter number to filename
    chapter_map = {}
    for chapter_file in chapter_files:
        match = re.search(r'chapter(\d+)', chapter_file.stem)
        if match:
            chapter_num = int(match.group(1))
            chapter_map[chapter_num] = chapter_file.stem
    
    for entry in toc_entries:
        chapter_num = entry['number']
        chapter_title = entry['title']
        
        if chapter_num in chapter_map:
            # Create link to chapter
            toc_html += f"""
            <li>
                <div class="toc-entry">
                    <a href="#{chapter_map[chapter_num]}">
                        <span class="chapter-number">{chapter_num}.</span>
                        <span class="chapter-title">{chapter_title}</span>
                    </a>
                    <span class="dots"></span>
                    <span class="page-number">{chapter_num}</span>
                </div>
                <ul>
            """
            
            # Add links to sections
            for i, section in enumerate(entry['sections']):
                section_num = i + 1
                toc_html += f"""
                <li class="section-entry">
                    <div class="toc-entry">
                        <a href="#{chapter_map[chapter_num]}_section{section_num}">{section}</a>
                        <span class="dots"></span>
                        <span class="page-number">{chapter_num}.{section_num}</span>
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

def create_navigation_links_html(toc_entries: List[Dict], chapter_files: List[Path]) -> str:
    """Create navigation links between chapters."""
    # Create navigation HTML
    nav_html = """
    <div class="navigation-links">
    """
    
    # Create a mapping of chapter number to filename and title
    chapter_map = {}
    for chapter_file in chapter_files:
        match = re.search(r'chapter(\d+)', chapter_file.stem)
        if match:
            chapter_num = int(match.group(1))
            chapter_map[chapter_num] = chapter_file.stem
    
    # Add navigation links for each chapter
    for entry in toc_entries:
        chapter_num = entry['number']
        if chapter_num in chapter_map:
            prev_chapter = chapter_num - 1
            next_chapter = chapter_num + 1
            
            nav_html += f'<div id="{chapter_map[chapter_num]}_nav" class="chapter-nav">\n'
            
            # Previous chapter link
            if prev_chapter in chapter_map:
                nav_html += f'<a href="#{chapter_map[prev_chapter]}" class="prev-link">← Previous Chapter</a>\n'
            
            # Next chapter link
            if next_chapter in chapter_map:
                nav_html += f'<a href="#{chapter_map[next_chapter]}" class="next-link">Next Chapter →</a>\n'
            
            nav_html += '</div>\n'
    
    nav_html += """
    </div>
    """
    
    return nav_html

def create_toc_pdf(toc_entries: List[Dict], chapter_files: List[Path]) -> Path:
    """Create a PDF with the table of contents with links to chapters."""
    # Generate TOC HTML
    toc_html = create_toc_html(toc_entries, chapter_files)
    
    # Save HTML file
    html_path = save_html_file(toc_html, "Table of Contents", "contents.html")
    
    # Convert HTML to PDF
    pdf_path = TEMP_DIR / "contents.pdf"
    html_to_pdf(html_path, pdf_path)
    
    logger.info(f"Created table of contents PDF: {pdf_path}")
    return pdf_path

def calculate_page_numbers(pdfs: List[Path]) -> Dict[str, int]:
    """Calculate starting page numbers for each chapter."""
    page_numbers = {}
    current_page = 1
    
    for pdf_path in pdfs:
        if "chapter" in pdf_path.stem:
            match = re.search(r'chapter(\d+)', pdf_path.stem)
            if match:
                chapter_num = int(match.group(1))
                page_numbers[chapter_num] = current_page
        
        # Update current page for next PDF
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            page_count = len(pdf_reader.pages)
            current_page += page_count
    
    return page_numbers

def create_bookmarks(toc_entries: List[Dict], page_numbers: Dict[int, int]) -> List[Dict]:
    """Create PDF bookmarks structure based on TOC entries and page numbers."""
    bookmarks = []
    
    # Create bookmarks with correct page numbers
    for entry in toc_entries:
        chapter_num = entry['number']
        
        if chapter_num in page_numbers:
            # Add chapter bookmark
            chapter_page = page_numbers[chapter_num]
            chapter_bookmark = {
                'title': f"Chapter {chapter_num}: {entry['title']}",
                'page': chapter_page,
                'children': []
            }
            bookmarks.append(chapter_bookmark)
            
            # Add section bookmarks
            section_offset = 0
            for section in entry['sections']:
                section_bookmark = {
                    'title': section,
                    'page': chapter_page + section_offset,
                    'children': []
                }
                chapter_bookmark['children'].append(section_bookmark)
                section_offset += 1  # Approximate section pages
    
    return bookmarks

def merge_pdfs_with_navigation(pdf_files: List[Path], output_path: Path, bookmarks: List[Dict]) -> None:
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
    """Convert front matter markdown files to PDFs."""
    front_matter_pdfs = []
    
    # Convert about_book.md if it exists
    if about_book_path.exists():
        html_content, _ = convert_md_to_html(about_book_path)
        html_path = save_html_file(html_content, "About the Book", "about_book.html")
        pdf_path = TEMP_DIR / "about_book.pdf"
        html_to_pdf(html_path, pdf_path)
        front_matter_pdfs.append(pdf_path)
    
    # Convert about_author.md if it exists
    if about_author_path.exists():
        html_content, _ = convert_md_to_html(about_author_path)
        html_path = save_html_file(html_content, "About the Author", "about_author.html")
        pdf_path = TEMP_DIR / "about_author.pdf"
        html_to_pdf(html_path, pdf_path)
        front_matter_pdfs.append(pdf_path)
    
    return front_matter_pdfs

def clean_up_temp_files():
    """Remove temporary files."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        logger.info(f"Removed temporary directory: {TEMP_DIR}")

def main():
    """Main function to convert markdown chapters to PDF and merge them with navigation."""
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
        chapter_nums = {}
        
        for md_file in chapter_files:
            pdf_file, chapter_num = convert_chapter_to_pdf(md_file)
            chapter_pdf_files.append(pdf_file)
            if chapter_num is not None:
                chapter_nums[pdf_file] = chapter_num
        
        # Create table of contents PDF
        toc_pdf = create_toc_pdf(toc_entries, chapter_files)
        
        # Calculate page numbers for each chapter
        all_pdfs = front_matter_pdfs + [toc_pdf] + chapter_pdf_files
        page_numbers = {}
        
        # Calculate page offsets
        current_page = 1
        for pdf_file in all_pdfs:
            if pdf_file in chapter_nums:
                chapter_num = chapter_nums[pdf_file]
                page_numbers[chapter_num] = current_page
            
            # Update page count
            with open(pdf_file, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                page_count = len(pdf_reader.pages)
                current_page += page_count
        
        # Create bookmarks
        bookmarks = create_bookmarks(toc_entries, page_numbers)
        
        # Merge all PDFs with navigation
        output_path = OUTPUT_DIR / "book_with_navigation.pdf"
        merge_pdfs_with_navigation(all_pdfs, output_path, bookmarks)
        
        logger.info(f"Book successfully created with navigation: {output_path}")
        
    except Exception as e:
        logger.exception(f"Error during conversion process: {e}")
        raise
    finally:
        # Clean up temporary files
        clean_up_temp_files()

if __name__ == "__main__":
    main()

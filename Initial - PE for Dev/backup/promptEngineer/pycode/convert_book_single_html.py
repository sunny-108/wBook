#!/usr/bin/env python
"""
Script to convert Markdown chapter files to a single HTML file then to PDF with working navigation.
This approach solves the navigation problem by creating a single HTML document with all anchors
before converting to PDF, rather than merging separate PDFs which breaks internal links.
"""

import os
import re
import sys
import tempfile
import shutil
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('md2pdf_single_html')

try:
    import markdown
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
except ImportError:
    logger.error("Required packages not installed. Please run: pip install markdown weasyprint")
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
    counter-reset: chapter;
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
    page-break-before: avoid;
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
}

.toc-entry {
    display: flex;
    width: 100%;
    align-items: baseline;
}

.chapter-number {
    font-weight: bold;
    margin-right: 0.5em;
}

.chapter-title {
    font-weight: bold;
}

.dots {
    flex-grow: 1;
    margin: 0 0.5em;
    border-bottom: 1px dotted #999;
    position: relative;
    bottom: 0.3em;
}

.page-number {
    font-weight: normal;
}

.section-entry {
    font-weight: normal;
    margin-top: 0.3em;
}

.section-entry .toc-entry {
    font-weight: normal;
}

/* Link styling */
a {
    color: #3498db;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Code blocks */
pre {
    background-color: #f8f8f8;
    border: 1px solid #ddd;
    border-radius: 3px;
    padding: 10px;
    font-family: "Courier New", monospace;
    font-size: 9pt;
    white-space: pre-wrap;
    overflow-wrap: break-word;
    overflow-x: auto;
    max-width: 100%;
}

code {
    font-family: "Courier New", monospace;
    font-size: 9pt;
    background-color: #f8f8f8;
    padding: 2px 4px;
    border-radius: 3px;
}

/* Table styling */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
}

table, th, td {
    border: 1px solid #ddd;
}

td, th {
    padding: 8px;
    text-align: left;
}

th {
    background-color: #f2f2f2;
}

img {
    max-width: 100%;
}

/* Front matter styling */
.front-matter {
    page-break-after: always;
}

.front-matter h1 {
    page-break-before: avoid;
}

/* Cover page */
.cover {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 80vh;
    text-align: center;
}

.cover h1 {
    font-size: 32pt;
    margin-bottom: 1em;
    color: #2c3e50;
    page-break-before: avoid;
}

.cover .subtitle {
    font-size: 18pt;
    margin-bottom: 2em;
    color: #7f8c8d;
}

/* Navigation */
.navigation {
    display: flex;
    justify-content: space-between;
    margin-top: 2em;
    border-top: 1px solid #eee;
    padding-top: 1em;
}

.prev-chapter, .next-chapter {
    padding: 5px 10px;
}

.prev-chapter:before {
    content: "← ";
}

.next-chapter:after {
    content: " →";
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

def convert_md_to_html(md_file: Path, chapter_num: Optional[int] = None) -> str:
    """Convert markdown file to HTML with proper anchors."""
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
    
    # Add ID to chapter heading (h1)
    if chapter_num is not None:
        # Replace the first h1 tag with one that has an id
        html_content = re.sub(
            r'<h1>(.*?)</h1>',
            f'<h1 id="chapter{chapter_num}">\\1</h1>',
            html_content,
            count=1
        )
        
        # Replace the h2 tags with ones that have ids
        section_num = 1
        
        def add_section_id(match):
            nonlocal section_num
            section_id = f"chapter{chapter_num}_section{section_num}"
            section_num += 1
            return f'<h2 id="{section_id}">{match.group(1)}</h2>'
        
        html_content = re.sub(r'<h2>(.*?)</h2>', add_section_id, html_content)
    
    return html_content

def create_toc_html(toc_entries: List[Dict]) -> str:
    """Create HTML for table of contents with links to chapters."""
    toc_html = """
    <div class="toc">
        <h1>Contents</h1>
        <ul>
    """
    
    for entry in toc_entries:
        chapter_num = entry['number']
        chapter_title = entry['title']
        
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

def create_cover_page_html(title: str = "Prompt Engineering for Developers", subtitle: str = "A Practical Guide"):
    """Create a cover page HTML."""
    cover_html = f"""
    <div class="cover">
        <h1>{title}</h1>
        <div class="subtitle">{subtitle}</div>
    </div>
    """
    return cover_html

def generate_complete_html() -> str:
    """Generate complete HTML document with all chapters and navigation."""
    chapter_files = get_chapter_files()
    toc_entries = parse_toc_from_chapter_flow()
    
    # Start with document structure
    complete_html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Complete Book</title>
    <style>{css}</style>
</head>
<body>
{cover}
"""
    
    # Add front matter if exists
    front_matter_html = ""
    
    # About the book
    about_book_path = WORKSPACE_DIR / "about_book.md"
    if about_book_path.exists():
        about_book_html = convert_md_to_html(about_book_path)
        front_matter_html += f'<div class="front-matter" id="about-book">{about_book_html}</div>'
    
    # About the author
    about_author_path = WORKSPACE_DIR / "about_author.md"
    if about_author_path.exists():
        about_author_html = convert_md_to_html(about_author_path)
        front_matter_html += f'<div class="front-matter" id="about-author">{about_author_html}</div>'
    
    complete_html = complete_html.format(
        css=CSS_STYLE,
        cover=create_cover_page_html()
    ) + front_matter_html
    
    # Add table of contents
    toc_html = create_toc_html(toc_entries)
    complete_html += toc_html
    
    # Add each chapter with its ID
    for chapter_file in chapter_files:
        match = re.search(r'chapter(\d+)', chapter_file.stem)
        chapter_num = int(match.group(1)) if match else None
        
        chapter_html = convert_md_to_html(chapter_file, chapter_num)
        chapter_div = f'<div class="chapter" id="chapter-content-{chapter_num}">{chapter_html}</div>'
        complete_html += chapter_div
        
        # Add navigation between chapters if it's not the last chapter
        if chapter_num is not None and chapter_num < len(chapter_files):
            next_chapter = chapter_num + 1
            prev_chapter = chapter_num - 1
            
            nav_html = '<div class="navigation">'
            
            # Previous chapter link (if not first)
            if prev_chapter > 0:
                nav_html += f'<a href="#chapter{prev_chapter}" class="prev-chapter">Previous Chapter</a>'
            else:
                nav_html += '<div></div>'  # Empty div for spacing
            
            # Next chapter link
            nav_html += f'<a href="#chapter{next_chapter}" class="next-chapter">Next Chapter</a>'
            
            nav_html += '</div>'
            complete_html += nav_html
    
    # Close HTML tags
    complete_html += """
</body>
</html>
"""
    
    return complete_html

def save_html_file(html_content: str, filename: str) -> Path:
    """Save HTML content to a file."""
    ensure_directories_exist()
    html_path = TEMP_HTML_DIR / filename
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Saved HTML to {html_path}")
    return html_path

def html_to_pdf(html_path: Path, pdf_path: Path) -> Path:
    """Convert HTML file to PDF."""
    HTML(filename=str(html_path)).write_pdf(
        pdf_path,
        stylesheets=[CSS(string=CSS_STYLE)]
    )
    
    logger.info(f"Converted {html_path} to {pdf_path}")
    return pdf_path

def clean_up_temp_files():
    """Remove temporary files."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        logger.info(f"Removed temporary directory: {TEMP_DIR}")

def main():
    """Main function to convert markdown chapters to PDF with working navigation."""
    try:
        # Ensure output directories exist
        ensure_directories_exist()
        
        # Generate complete HTML with all chapters and TOC
        complete_html = generate_complete_html()
        
        # Save HTML file
        html_path = save_html_file(complete_html, "complete_book.html")
        
        # Convert HTML to PDF
        pdf_path = OUTPUT_DIR / "book_with_working_navigation.pdf"
        html_to_pdf(html_path, pdf_path)
        
        logger.info(f"Book successfully created with navigation: {pdf_path}")
        
    except Exception as e:
        logger.exception(f"Error during conversion process: {e}")
        raise
    finally:
        # Clean up temporary files
        clean_up_temp_files()

if __name__ == "__main__":
    main()

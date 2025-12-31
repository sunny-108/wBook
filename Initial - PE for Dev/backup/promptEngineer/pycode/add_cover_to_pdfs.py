#!/usr/bin/env python
"""
Script to add a book cover and title page to existing PDFs in the output directory.
"""

import os
import sys
import glob
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('add_cover_page')

try:
    from weasyprint import HTML, CSS
    from PyPDF2 import PdfMerger
except ImportError:
    logger.error("Required packages not installed. Please run: pip install weasyprint PyPDF2")
    sys.exit(1)

# Directory paths
WORKSPACE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = WORKSPACE_DIR / "output"
COVER_DIR = WORKSPACE_DIR / "cover_design"
TEMP_DIR = OUTPUT_DIR / "temp"

# Ensure directories exist
TEMP_DIR.mkdir(exist_ok=True)

# CSS for the cover page
COVER_CSS = """
@page {
    size: 6in 9in;
    margin: 0;
}

body {
    margin: 0;
    padding: 0;
    width: 6in;
    height: 9in;
    background: linear-gradient(to top, #0A2342, #2C74B3, #483D8B);
}

.cover {
    width: 100%;
    height: 100%;
    padding: 0.5in;
    box-sizing: border-box;
    position: relative;
}

.title-container {
    text-align: center;
    margin-top: 1.5in;
    position: relative;
    z-index: 10;
}

.main-title {
    color: white;
    font-family: 'Arial', sans-serif;
    font-size: 36pt;
    font-weight: 800;
    margin: 0;
    line-height: 1.2;
    letter-spacing: 1px;
    text-shadow: 0px 2px 4px rgba(0,0,0,0.5);
}

.subtitle {
    color: #B5D0FF;
    font-family: 'Arial', sans-serif;
    font-size: 18pt;
    font-weight: 300;
    margin-top: 15px;
    margin-bottom: 0;
}

.graphic {
    margin: 0.8in auto;
    height: 3in;
    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
}

.code-column {
    background-color: rgba(30, 30, 30, 0.7);
    border-radius: 5px;
    padding: 15px;
    width: 1.8in;
    font-family: 'Courier New', monospace;
    font-size: 11pt;
    color: #d4d4d4;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    z-index: 10;
    position: relative;
    left: -30px;
}

.prompt-column {
    background-color: rgba(49, 49, 68, 0.7);
    border-radius: 5px;
    padding: 15px;
    width: 1.8in;
    font-family: 'Arial', sans-serif;
    font-size: 11pt;
    color: #e0e0e0;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    z-index: 10;
    position: relative;
    right: -30px;
}

.connector-graphic {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 5;
}

.keyword { color: #569CD6; }
.string { color: #CE9178; }
.comment { color: #6A9955; }
.function { color: #DCDCAA; }
.variable { color: #9CDCFE; }

.author {
    text-align: center;
    position: absolute;
    bottom: 0.8in;
    left: 0;
    width: 100%;
    color: white;
    font-family: 'Arial', sans-serif;
    font-size: 14pt;
    font-weight: 300;
    z-index: 10;
}

.brackets {
    position: absolute;
    font-size: 120pt;
    font-weight: 300;
    color: rgba(255,255,255,0.1);
    font-family: 'Courier New', monospace;
}

.bracket-left {
    top: 40%;
    left: 15px;
}

.bracket-right {
    top: 40%;
    right: 15px;
}

.connection-dots {
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    z-index: 5;
}

.dot {
    width: 8px;
    height: 8px;
    background-color: #8A2BE2;
    border-radius: 50%;
    position: absolute;
}

.connection-line {
    position: absolute;
    height: 2px;
    background: linear-gradient(to right, transparent, #00FFFF, transparent);
}
"""

# CSS for the title page
TITLE_PAGE_CSS = """
@page {
    size: 6in 9in;
    margin: 0;
}

body {
    margin: 0;
    padding: 0;
    width: 6in;
    height: 9in;
    background-color: white;
    font-family: 'Arial', sans-serif;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 1in;
    box-sizing: border-box;
}

.title-section {
    text-align: center;
    margin-top: 1in;
}

.book-title {
    font-size: 24pt;
    font-weight: 700;
    color: #0A2342;
    margin-bottom: 0.2in;
}

.book-subtitle {
    font-size: 18pt;
    font-weight: 400;
    color: #2C74B3;
}

.author-section {
    text-align: center;
    margin-bottom: 1in;
}

.author-name {
    font-size: 16pt;
    color: #333;
}
"""

# HTML for the cover
COVER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Book Cover</title>
</head>
<body>
    <div class="cover">
        <div class="title-container">
            <h1 class="main-title">PROMPT ENGINEERING</h1>
            <h2 class="main-title" style="font-size: 28pt;">FOR DEVELOPERS</h2>
            <p class="subtitle">Crafting Intelligent LLM Solutions</p>
        </div>
        
        <div class="graphic">
            <div class="code-column">
                <span class="keyword">def</span> <span class="function">prompt_for_code</span>():
                <br>&nbsp;&nbsp;<span class="variable">result</span> = <span class="string">"hello"</span>
                <br>&nbsp;&nbsp;<span class="keyword">return</span> <span class="variable">result</span>
            </div>
            
            <div class="connection-dots">
                <!-- Connection dots between code and natural language -->
                <div class="connection-line" style="width: 100px; top: 50px; left: -50px;"></div>
                <div class="dot" style="top: 50px; left: -40px;"></div>
                <div class="dot" style="top: 50px; left: -20px;"></div>
                <div class="dot" style="top: 50px; left: 0px;"></div>
                <div class="dot" style="top: 50px; left: 20px;"></div>
                <div class="dot" style="top: 50px; left: 40px;"></div>
                
                <div class="connection-line" style="width: 100px; top: 100px; left: -50px;"></div>
                <div class="dot" style="top: 100px; left: -40px;"></div>
                <div class="dot" style="top: 100px; left: -20px;"></div>
                <div class="dot" style="top: 100px; left: 0px;"></div>
                <div class="dot" style="top: 100px; left: 20px;"></div>
                <div class="dot" style="top: 100px; left: 40px;"></div>
                
                <div class="connection-line" style="width: 100px; top: 150px; left: -50px;"></div>
                <div class="dot" style="top: 150px; left: -40px;"></div>
                <div class="dot" style="top: 150px; left: -20px;"></div>
                <div class="dot" style="top: 150px; left: 0px;"></div>
                <div class="dot" style="top: 150px; left: 20px;"></div>
                <div class="dot" style="top: 150px; left: 40px;"></div>
            </div>
            
            <div class="prompt-column">
                Write a Python function that:
                <br>1. Takes a string input
                <br>2. Processes the text
                <br>3. Returns a greeting
            </div>
        </div>
        
        <div class="author">By Sunny Shivam</div>
        
        <div class="brackets bracket-left">{</div>
        <div class="brackets bracket-right">}</div>
    </div>
</body>
</html>
"""

# HTML for the title page
TITLE_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Title Page</title>
</head>
<body>
    <div class="title-section">
        <div class="book-title">PROMPT ENGINEERING FOR DEVELOPERS</div>
        <div class="book-subtitle">Crafting Intelligent LLM Solutions</div>
    </div>
    
    <div class="author-section">
        <div class="author-name">Sunny Shivam</div>
    </div>
</body>
</html>
"""

def create_cover_pdf():
    """Create a PDF of the book cover."""
    cover_pdf_path = TEMP_DIR / "cover.pdf"
    
    # Use WeasyPrint to convert HTML to PDF
    HTML(string=COVER_HTML).write_pdf(
        cover_pdf_path,
        stylesheets=[CSS(string=COVER_CSS)]
    )
    
    logger.info(f"Created cover PDF: {cover_pdf_path}")
    return cover_pdf_path

def create_title_page_pdf():
    """Create a PDF of the title page."""
    title_page_pdf_path = TEMP_DIR / "title_page.pdf"
    
    # Use WeasyPrint to convert HTML to PDF
    HTML(string=TITLE_PAGE_HTML).write_pdf(
        title_page_pdf_path,
        stylesheets=[CSS(string=TITLE_PAGE_CSS)]
    )
    
    logger.info(f"Created title page PDF: {title_page_pdf_path}")
    return title_page_pdf_path

def add_pages_to_pdfs():
    """Add cover and title page to all PDFs in the output directory."""
    # Create the cover and title page PDFs
    cover_pdf_path = create_cover_pdf()
    title_page_pdf_path = create_title_page_pdf()
    
    # Find all PDFs in the output directory (excluding temporary files)
    pdf_files = [f for f in OUTPUT_DIR.glob("*.pdf") if "temp" not in f.name]
    
    for pdf_file in pdf_files:
        # Create a new output filename with "_with_cover" suffix
        output_file = OUTPUT_DIR / f"{pdf_file.stem}_with_cover.pdf"
        
        # Create a PDF merger object
        merger = PdfMerger()
        
        # Add the cover page
        merger.append(str(cover_pdf_path))
        
        # Add the title page
        merger.append(str(title_page_pdf_path))
        
        # Add the original PDF
        merger.append(str(pdf_file))
        
        # Write the output file
        merger.write(str(output_file))
        merger.close()
        
        logger.info(f"Added cover and title page to {pdf_file}, saved as {output_file}")

if __name__ == "__main__":
    try:
        add_pages_to_pdfs()
        print("Successfully added cover and title page to all PDFs in the output directory.")
    except Exception as e:
        logger.exception(f"Error adding cover and title page: {e}")
        print(f"Error: {e}")
        sys.exit(1)

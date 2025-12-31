#!/usr/bin/env python3
"""
HTML to PDF Converter for eBook
Converts ebook.html to a properly formatted PDF document
"""

import os
import sys
from pathlib import Path
import weasyprint
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

def convert_html_to_pdf():
    """
    Convert ebook.html to PDF format with proper styling
    """
    # Define paths
    script_dir = Path(__file__).parent
    html_file = script_dir / "htmlFiles" / "ebook.html"
    output_dir = script_dir / "output"
    pdf_file = output_dir / "ebook.pdf"
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Check if HTML file exists
    if not html_file.exists():
        print(f"Error: HTML file not found at {html_file}")
        return False
    
    print(f"Converting {html_file} to PDF...")
    print(f"Output will be saved to {pdf_file}")
    
    try:
        # Configure fonts for better PDF rendering
        font_config = FontConfiguration()
        
        # Read HTML content
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Create HTML object with base URL for relative paths
        html_doc = HTML(string=html_content, base_url=str(html_file.parent))
        
        # Additional CSS for PDF optimization
        pdf_css = CSS(string="""
            /* Override page numbering - let HTML CSS handle it */
            @page {
                size: letter;
                margin: 2cm;
            }
            
            @page main {
                @bottom-center {
                    content: "Page " counter(main-page);
                    font-size: 10pt;
                    color: #666;
                }
            }
            
            @page cover {
                @bottom-center { content: none; }
            }
            
            @page title {
                @bottom-center { content: none; }
            }
            
            @page front {
                @bottom-center { content: none; }
            }
            
            @page toc {
                @bottom-center { content: none; }
            }
            
            /* Force page counter reset for main content */
            body {
                counter-reset: page;
            }
            
            .book-content {
                counter-reset: main-page 1;
            }
            
            /* Ensure proper page breaks */
            .chapter {
                page-break-before: always;
                page: main;
                counter-increment: main-page;
            }
            
            .front-matter {
                page-break-after: always;
                page: front;
            }
            
            .toc {
                page-break-before: always;
                page-break-after: always;
                page: toc;
            }
            
            .cover-page {
                page: cover;
            }
            
            .title-page {
                page: title;
            }
            
            /* Cover page styling for PDF */
            .cover-page {
                page-break-after: always;
                width: 100%;
                height: 100vh;
                background: linear-gradient(to top, #0A2342, #2C74B3, #483D8B);
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
            
            /* Title page styling for PDF */
            .title-cover {
                page-break-after: always;
                width: 100%;
                height: 100vh;
                background: white;
                padding: 0.5in;
                box-sizing: border-box;
                position: relative;
                text-align: center;
            }
            
            .book-title1 {
                margin-top: 2in;
                padding: 0 1in;
                text-align: center;
            }
            
            .main-title1 {
                font-size: 42pt;
                font-weight: bold;
                color: #2c3e50;
                margin: 0;
                line-height: 1.2;
                letter-spacing: 1px;
            }
            
            .subtitle1 {
                font-size: 24pt;
                color: #3498db;
                font-weight: 300;
                margin: 20px 0 0 0;
            }
            
            .author-section1 {
                position: absolute;
                bottom: 2in;
                left: 0;
                right: 0;
                padding: 0 1in;
                text-align: center;
            }
            
            .author-name1 {
                font-size: 24pt;
                color: #2c3e50;
                font-weight: 400;
                margin: 0;
            }
            
            .title-content {
                display: flex;
                flex-direction: column;
                height: 100%;
            }
            
            .book-title-section {
                margin-top: 3in;
                margin-bottom: auto;
            }
            
            .title-main {
                font-size: 42pt;
                font-weight: bold;
                color: #2c3e50;
                margin: 0 0 2em 0;
                line-height: 1.2;
                letter-spacing: 1px;
            }
            
            .title-subtitle {
                font-size: 24pt;
                color: #3498db;
                font-weight: 300;
                margin: 0;
            }
            
            .author-bottom {
                margin-top: auto;
                margin-bottom: 1in;
            }
            
            .title-author {
                font-size: 24pt;
                color: #2c3e50;
                font-weight: 400;
                margin: 0;
            }
            
            /* Improve code block rendering */
            pre, code {
                font-family: "Courier New", "Liberation Mono", monospace;
                font-size: 9pt;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            
            /* Better table rendering */
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 1em 0;
            }
            
            table, th, td {
                border: 1px solid #ddd;
            }
            
            th, td {
                padding: 8px;
                text-align: left;
                vertical-align: top;
            }
            
            /* Ensure images fit properly */
            img {
                max-width: 100%;
                height: auto;
            }
            
            /* Navigation elements - hide in print */
            .navigation {
                display: none;
            }
        """, font_config=font_config)
        
        # Convert to PDF
        html_doc.write_pdf(
            str(pdf_file),
            stylesheets=[pdf_css],
            font_config=font_config,
            optimize_images=True
        )
        
        print(f"✅ Successfully converted to PDF: {pdf_file}")
        print(f"📄 File size: {pdf_file.stat().st_size / 1024 / 1024:.2f} MB")
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        return False

def install_dependencies():
    """
    Install required dependencies if not available
    """
    try:
        import weasyprint
        print("✅ WeasyPrint is already installed")
        return True
    except ImportError:
        print("📦 WeasyPrint not found. Please install it first.")
        print("Run the following commands:")
        print("  python3 -m venv venv")
        print("  source venv/bin/activate")
        print("  pip install weasyprint")
        print("  python3 convert_to_pdf.py")
        print("\nOr use the provided convert.sh script")
        return False

def main():
    """
    Main function to handle the conversion process
    """
    print("🔄 HTML to PDF Converter for eBook")
    print("=" * 40)
    
    # Check and install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Perform conversion
    success = convert_html_to_pdf()
    
    if success:
        print("\n🎉 Conversion completed successfully!")
        print("You can find your PDF in the 'output' directory.")
    else:
        print("\n💥 Conversion failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

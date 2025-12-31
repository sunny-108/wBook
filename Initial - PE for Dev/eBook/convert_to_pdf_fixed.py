#!/usr/bin/env python3
"""
Fixed PDF Generator with Manual Page Number Control
Converts ebook.html to PDF with proper page numbering starting from Chapter 1
"""

import os
import sys
from pathlib import Path
import weasyprint
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

def convert_html_to_pdf_fixed():
    """
    Convert ebook.html to PDF format with manually controlled page numbering
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
    
    print(f"🔄 Converting {html_file} to PDF with fixed page numbering...")
    print(f"Output will be saved to {pdf_file}")
    
    try:
        # Configure fonts for better PDF rendering
        font_config = FontConfiguration()
        
        # Read HTML content
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Create HTML object with base URL for relative paths
        html_doc = HTML(string=html_content, base_url=str(html_file.parent))
        
        # CSS that completely removes automatic page numbering
        # We'll add them manually later
        pdf_css = CSS(string="""
            @page {
                size: letter;
                margin: 2cm;
                @bottom-center {
                    content: none;
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
            
            @page main {
                @bottom-center { content: none; }
            }
            
            /* Ensure proper page breaks */
            .chapter {
                page-break-before: always;
                page: main;
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
                page-break-after: always;
            }
            
            .title-page {
                page: title;
                page-break-after: always;
            }
        """)
        
        # Generate PDF without page numbers first
        print("🔧 Generating base PDF...")
        pdf_bytes = html_doc.write_pdf(
            stylesheets=[pdf_css],
            font_config=font_config,
            optimize_size=('fonts', 'images')
        )
        
        if pdf_bytes is None:
            print("❌ Failed to generate PDF")
            return False
        
        # Write to file
        with open(pdf_file, 'wb') as f:
            f.write(pdf_bytes)
        
        print(f"✅ Successfully converted to PDF: {pdf_file}")
        print(f"📄 File size: {len(pdf_bytes) / (1024*1024):.2f} MB")
        
        # Now we need to add page numbers manually to main content pages
        print("🔧 Adding page numbers to main content...")
        add_page_numbers_to_main_content(pdf_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

def add_page_numbers_to_main_content(pdf_path):
    """
    Add page numbers starting from 1 to the main content pages only
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from PyPDF2 import PdfReader, PdfWriter
        import io
        
        # Read the existing PDF
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        
        # Find where Chapter 1 starts
        chapter1_page = None
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if "Chapter 1:" in text and "Introduction to Prompt Engineering" in text:
                chapter1_page = page_num
                break
        
        if chapter1_page is None:
            print("❌ Could not find Chapter 1 in PDF")
            return False
        
        print(f"📍 Chapter 1 found at page index {chapter1_page} (page {chapter1_page + 1} in PDF)")
        
        # Process each page
        for page_num, page in enumerate(reader.pages):
            if page_num >= chapter1_page:  # Main content pages
                # Create a new PDF with just the page number
                packet = io.BytesIO()
                can = canvas.Canvas(packet, pagesize=letter)
                
                # Calculate page number for main content (starts from 1)
                main_page_num = page_num - chapter1_page + 1
                
                # Add page number at bottom center
                can.setFont("Helvetica", 10)
                can.setFillGray(0.4)  # Gray color
                can.drawCentredString(letter[0]/2, 50, f"Page {main_page_num}")
                can.save()
                
                # Move to the beginning of the StringIO buffer
                packet.seek(0)
                new_pdf = PdfReader(packet)
                
                # Merge the page number with the original page
                page.merge_page(new_pdf.pages[0])
            
            # Add the page (with or without page number) to the output
            writer.add_page(page)
        
        # Write the modified PDF
        temp_path = pdf_path.with_suffix('.temp.pdf')
        with open(temp_path, 'wb') as output_file:
            writer.write(output_file)
        
        # Replace the original file
        os.replace(temp_path, pdf_path)
        
        print(f"✅ Page numbers added starting from Chapter 1")
        return True
        
    except Exception as e:
        print(f"❌ Error adding page numbers: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = convert_html_to_pdf_fixed()
    if success:
        print("\n🎉 Conversion completed successfully!")
        print("📖 Chapter 1 now starts at Page 1 in the PDF!")
        print("You can find your PDF in the 'output' directory.")
    else:
        print("\n❌ Conversion failed!")
        sys.exit(1)

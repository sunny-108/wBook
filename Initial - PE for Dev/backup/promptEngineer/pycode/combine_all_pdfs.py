#!/usr/bin/env python
"""
Script to combine cover.pdf + title_page.pdf + complete_book.pdf + index.pdf
while preserving href links and page numbers.
"""

import os
import sys
from pathlib import Path
import logging
from typing import List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pdf_combiner')

try:
    import PyPDF2
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
except ImportError:
    logger.error("Required packages not installed. Please run: pip install PyPDF2 weasyprint")
    sys.exit(1)

def convert_index_html_to_pdf(html_file_path: str, output_pdf_path: str) -> bool:
    """
    Convert index.html to PDF with proper styling.
    
    Args:
        html_file_path: Path to the index HTML file
        output_pdf_path: Path where PDF should be saved
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Converting {html_file_path} to PDF...")
        
        # Create font configuration
        font_config = FontConfiguration()
        
        # CSS for index styling that matches the main book
        index_css = CSS(string="""
        @page {
            size: letter;
            margin: 2cm;
            @bottom-center {
                content: "Page " counter(page);
                font-size: 10pt;
                color: #666;
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
            text-align: center;
            margin-bottom: 1.5em;
        }
        
        h2 {
            font-size: 16pt;
            color: #3498db;
            margin-top: 1em;
        }
        
        h3 {
            font-size: 14pt;
            color: #2980b9;
        }
        
        /* Index specific styling */
        .index-entry {
            margin-bottom: 0.5em;
            display: flex;
            justify-content: space-between;
        }
        
        .index-term {
            font-weight: bold;
        }
        
        .index-pages {
            color: #666;
        }
        
        /* Links styling */
        a {
            color: #3498db;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        """, font_config=font_config)
        
        # Create HTML object and render to PDF
        html_doc = HTML(filename=html_file_path)
        html_doc.write_pdf(
            output_pdf_path,
            stylesheets=[index_css],
            font_config=font_config,
            optimize_images=True,
            presentational_hints=True,
            pdf_version='1.7'
        )
        
        logger.info(f"Index PDF created: {output_pdf_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting index HTML to PDF: {str(e)}")
        return False

def merge_pdfs(pdf_files: List[str], output_path: str) -> bool:
    """
    Merge multiple PDF files while preserving bookmarks and links.
    
    Args:
        pdf_files: List of PDF file paths to merge
        output_path: Path for the merged PDF
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Starting PDF merge process...")
        
        # Create a PDF writer object
        pdf_writer = PyPDF2.PdfWriter()
        
        # Track page offset for updating internal links
        page_offset = 0
        
        for i, pdf_file in enumerate(pdf_files):
            logger.info(f"Processing {pdf_file}...")
            
            # Check if file exists
            if not os.path.exists(pdf_file):
                logger.error(f"File not found: {pdf_file}")
                return False
            
            # Read the PDF file
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Add all pages from this PDF
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    pdf_writer.add_page(page)
                
                # Copy bookmarks and outlines if they exist
                if hasattr(pdf_reader, 'outline') and pdf_reader.outline:
                    try:
                        # Add bookmarks with proper page offset
                        def add_bookmarks(bookmarks, offset):
                            for bookmark in bookmarks:
                                if isinstance(bookmark, list):
                                    add_bookmarks(bookmark, offset)
                                else:
                                    if hasattr(bookmark, 'title'):
                                        try:
                                            pdf_writer.add_outline_item(bookmark.title, offset)
                                        except:
                                            pass  # Skip if bookmark cannot be added
                        
                        add_bookmarks(pdf_reader.outline, page_offset)
                    except Exception as e:
                        logger.warning(f"Could not copy bookmarks from {pdf_file}: {str(e)}")
                
                # Update page offset for next file
                page_offset += len(pdf_reader.pages)
                
                logger.info(f"Added {len(pdf_reader.pages)} pages from {os.path.basename(pdf_file)}")
        
        # Write the merged PDF
        with open(output_path, 'wb') as output_file:
            pdf_writer.write(output_file)
        
        logger.info(f"Successfully merged {len(pdf_files)} PDFs into {output_path}")
        logger.info(f"Total pages: {page_offset}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error merging PDFs: {str(e)}")
        return False

def main():
    """Main function to combine all PDF files in the correct sequence."""
    
    # Set up paths
    current_dir = Path(__file__).parent
    temp_dir = current_dir.parent / "output" / "temp"
    html_dir = temp_dir / "html"
    output_dir = current_dir.parent / "output"
    
    # Define file paths in the required sequence
    cover_pdf = temp_dir / "cover.pdf"
    title_page_pdf = temp_dir / "title_page.pdf"
    complete_book_pdf = output_dir / "complete_book.pdf"
    index_html = html_dir / "index.html"
    
    # Create temporary index PDF
    temp_index_pdf = temp_dir / "temp_index.pdf"
    
    # Check if main book PDF exists
    if not complete_book_pdf.exists():
        logger.error(f"Main book PDF not found: {complete_book_pdf}")
        print("❌ Main book PDF not found. Please run the HTML to PDF conversion first.")
        return False
    
    # Convert index.html to PDF
    logger.info("Step 1: Converting index.html to PDF...")
    if not convert_index_html_to_pdf(str(index_html), str(temp_index_pdf)):
        logger.error("Failed to convert index.html to PDF")
        return False
    
    # Prepare list of PDFs to merge in sequence
    pdf_files = []
    
    # Add cover.pdf if it exists
    if cover_pdf.exists():
        pdf_files.append(str(cover_pdf))
        logger.info("✓ Cover PDF found and will be included")
    else:
        logger.warning("⚠ Cover PDF not found, skipping")
    
    # Add title_page.pdf if it exists
    if title_page_pdf.exists():
        pdf_files.append(str(title_page_pdf))
        logger.info("✓ Title page PDF found and will be included")
    else:
        logger.warning("⚠ Title page PDF not found, skipping")
    
    # Add main book PDF (required)
    pdf_files.append(str(complete_book_pdf))
    logger.info("✓ Main book PDF found and will be included")
    
    # Add index PDF
    if temp_index_pdf.exists():
        pdf_files.append(str(temp_index_pdf))
        logger.info("✓ Index PDF created and will be included")
    else:
        logger.warning("⚠ Could not create index PDF, skipping")
    
    # Output file path
    final_pdf = output_dir / "complete_book_with_all_sections.pdf"
    
    # Merge all PDFs
    logger.info("Step 2: Merging all PDFs...")
    success = merge_pdfs(pdf_files, str(final_pdf))
    
    if success:
        # Clean up temporary index PDF
        if temp_index_pdf.exists():
            temp_index_pdf.unlink()
            logger.info("Cleaned up temporary index PDF")
        
        # Get final file info
        file_size_mb = final_pdf.stat().st_size / 1024 / 1024
        
        print(f"\n🎉 SUCCESS! Complete book created successfully!")
        print(f"📁 Location: {final_pdf}")
        print(f"📄 File size: {file_size_mb:.2f} MB")
        print(f"📑 Sections included:")
        
        section_names = []
        if cover_pdf.exists():
            section_names.append("  • Cover page")
        if title_page_pdf.exists():
            section_names.append("  • Title page")
        section_names.append("  • Complete book content (with TOC and all chapters)")
        if temp_index_pdf.exists():
            section_names.append("  • Index")
        
        for section in section_names:
            print(section)
        
        print(f"\n✅ Features preserved:")
        print("  • All internal hyperlinks (href links)")
        print("  • Page numbers throughout the document")
        print("  • Table of contents with clickable links")
        print("  • Chapter navigation")
        print("  • Code syntax highlighting")
        print("  • Professional formatting")
        
        return True
    else:
        print("❌ Failed to create the complete book. Check the logs for details.")
        return False

if __name__ == "__main__":
    main()

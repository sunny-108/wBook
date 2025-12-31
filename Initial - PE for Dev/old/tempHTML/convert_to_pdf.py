#!/usr/bin/env python3
"""
HTML to PDF Converter Script
Converts the complete_book.html file to a PDF document.
"""

import os
import sys
from pathlib import Path
import weasyprint
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

def convert_html_to_pdf(html_file_path, output_pdf_path):
    """
    Convert HTML file to PDF using WeasyPrint.
    
    Args:
        html_file_path (str): Path to the input HTML file
        output_pdf_path (str): Path for the output PDF file
    """
    try:
        # Check if input file exists
        if not os.path.exists(html_file_path):
            print(f"Error: HTML file '{html_file_path}' not found.")
            return False
        
        print(f"Converting '{html_file_path}' to PDF...")
        print(f"Output will be saved as: '{output_pdf_path}'")
        
        # Create font configuration for better font handling
        font_config = FontConfiguration()
        
        # Load HTML file and convert to PDF
        html_doc = HTML(filename=html_file_path)
        
        # Optional: Add custom CSS for better PDF formatting
        custom_css = CSS(string='''
            @page {
                size: letter;
                margin: 2cm;
                @bottom-center {
                    content: "Page " counter(page);
                }
            }
            
            /* Ensure proper page breaks for chapters */
            .chapter {
                page-break-before: always;
            }
            
            /* Improve table formatting for PDF */
            table {
                page-break-inside: avoid;
            }
            
            /* Ensure code blocks don't break awkwardly */
            pre, code {
                page-break-inside: avoid;
            }
            
            /* Better handling of TOC dots in PDF */
            .dots {
                border-bottom: 1px dotted #999;
            }
        ''', font_config=font_config)
        
        # Generate PDF
        html_doc.write_pdf(
            output_pdf_path,
            stylesheets=[custom_css],
            font_config=font_config
        )
        
        print(f"✅ Successfully converted to PDF: {output_pdf_path}")
        
        # Get file size for confirmation
        file_size = os.path.getsize(output_pdf_path)
        print(f"📄 PDF file size: {file_size / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        return False

def main():
    """Main function to handle command line arguments and run conversion."""
    
    # Set up paths
    script_dir = Path(__file__).parent
    html_file = script_dir / "html" / "complete_book.html"
    output_file = script_dir / "complete_book.pdf"
    
    # Allow command line arguments for custom paths
    if len(sys.argv) > 1:
        html_file = Path(sys.argv[1])
    
    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])
    
    print("HTML to PDF Converter")
    print("=" * 50)
    print(f"Input HTML: {html_file}")
    print(f"Output PDF: {output_file}")
    print("=" * 50)
    
    # Perform conversion
    success = convert_html_to_pdf(str(html_file), str(output_file))
    
    if success:
        print("\n🎉 Conversion completed successfully!")
        print(f"You can find your PDF at: {output_file}")
    else:
        print("\n💥 Conversion failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

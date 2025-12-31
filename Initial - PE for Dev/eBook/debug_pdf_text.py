#!/usr/bin/env python3
"""
Debug script to see what text is actually in the PDF pages
"""

import PyPDF2
from pathlib import Path

def debug_pdf_text():
    """Debug what text is in the PDF pages"""
    pdf_path = Path(__file__).parent / "output" / "ebook.pdf"
    
    if not pdf_path.exists():
        print(f"❌ PDF file not found at {pdf_path}")
        return False
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Find Chapter 1
            chapter1_page_index = None
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if "Chapter 1:" in text and "Introduction to Prompt Engineering" in text:
                    chapter1_page_index = page_num
                    break
            
            if chapter1_page_index is None:
                print("❌ Could not find Chapter 1")
                return False
            
            print(f"📍 Chapter 1 found at PDF page {chapter1_page_index + 1}")
            
            # Check the first few pages starting from Chapter 1
            for i in range(3):  # Check first 3 chapter pages
                page_index = chapter1_page_index + i
                if page_index < len(pdf_reader.pages):
                    page = pdf_reader.pages[page_index]
                    text = page.extract_text()
                    
                    print(f"\n=== PDF Page {page_index + 1} ===")
                    # Show last 200 characters to see if page number is at the end
                    print("Last 200 characters:")
                    print(repr(text[-200:]))
                    
                    # Look for any "Page" pattern
                    if "Page" in text:
                        start = text.find("Page")
                        print(f"Found 'Page' at position {start}:")
                        print(repr(text[start:start+20]))
                    else:
                        print("No 'Page' found in text")
            
            return True
            
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return False

if __name__ == "__main__":
    debug_pdf_text()

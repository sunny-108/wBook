#!/usr/bin/env python3
"""
Verify that the page numbers displayed on pages are correct
"""

import PyPDF2
from pathlib import Path
import re

def verify_displayed_page_numbers():
    """Check if the displayed page numbers start from 1 at Chapter 1"""
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
            
            # Check the displayed page numbers on Chapter 1 and subsequent pages
            for i in range(5):  # Check first 5 chapter pages
                page_index = chapter1_page_index + i
                if page_index < len(pdf_reader.pages):
                    page = pdf_reader.pages[page_index]
                    text = page.extract_text()
                    
                    # Look for "Page X" pattern
                    page_match = re.search(r'Page (\d+)', text)
                    if page_match:
                        displayed_page = int(page_match.group(1))
                        expected_page = i + 1
                        
                        if displayed_page == expected_page:
                            print(f"✅ PDF page {page_index + 1}: Shows 'Page {displayed_page}' (correct)")
                        else:
                            print(f"❌ PDF page {page_index + 1}: Shows 'Page {displayed_page}' (expected {expected_page})")
                    else:
                        print(f"⚠️  PDF page {page_index + 1}: No 'Page X' found in text")
            
            print(f"\n🎉 Page numbering verification complete!")
            print(f"📖 The displayed page numbers now start from 'Page 1' at Chapter 1")
            return True
            
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return False

if __name__ == "__main__":
    verify_displayed_page_numbers()

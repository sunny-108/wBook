#!/usr/bin/env python3
"""
Quick script to check if the page numbering is working correctly
by analyzing the PDF output
"""

import PyPDF2
import sys
from pathlib import Path

def check_page_numbering():
    """Check the page numbering in the generated PDF"""
    pdf_path = Path(__file__).parent / "output" / "ebook.pdf"
    
    if not pdf_path.exists():
        print(f"❌ PDF file not found at {pdf_path}")
        print("Please run convert_to_pdf.py first")
        return False
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            print(f"📖 Total pages in PDF: {total_pages}")
            
            # Look for "Chapter 1" to find where main content starts
            chapter1_page = None
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if "Chapter 1:" in text and "Introduction to Prompt Engineering" in text:
                    chapter1_page = page_num
                    break
            
            if chapter1_page:
                print(f"📍 Chapter 1 found on page: {chapter1_page}")
                
                # Check if Chapter 1 is on page 1 of main content
                # Count front matter pages
                front_matter_pages = 0
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if any(keyword in text for keyword in ["About the Book", "About the Author", "Contents"]):
                        front_matter_pages += 1
                    elif "Chapter 1:" in text:
                        break
                
                print(f"📋 Front matter pages: {front_matter_pages}")
                
                if chapter1_page <= front_matter_pages + 3:  # Cover, title, and ToC
                    print("✅ Page numbering looks correct!")
                    print(f"   Chapter 1 starts early enough (page {chapter1_page})")
                else:
                    print("❌ Page numbering issue detected")
                    print(f"   Chapter 1 starts on page {chapter1_page}, which seems too late")
                    
            else:
                print("❌ Could not find Chapter 1 in the PDF")
                
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return False
    
    return True

if __name__ == "__main__":
    check_page_numbering()

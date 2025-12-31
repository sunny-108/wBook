#!/usr/bin/env python3
"""
Debug HTML structure to check if title page is present
"""

from pathlib import Path
from bs4 import BeautifulSoup

def debug_html():
    script_dir = Path(__file__).parent
    html_file = script_dir / "htmlFiles" / "ebook.html"
    
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    
    # Find cover page
    cover_page = soup.find('div', class_='cover-page')
    print("Cover page found:", cover_page is not None)
    if cover_page:
        title = cover_page.find('h1')
        if title:
            print(f"Cover title: {title.get_text()}")
    
    # Find title page
    title_page = soup.find('div', class_='title-page')
    print("Title page found:", title_page is not None)
    if title_page:
        title = title_page.find('h1')
        if title:
            print(f"Title page title: {title.get_text()}")
        author = title_page.find('p', {'class': 'title-author'})
        if author:
            print(f"Title page author: {author.get_text()}")
    
    # Check page structure
    all_divs = soup.find_all('div', class_=['cover-page', 'title-page', 'book-content'])
    print(f"\nPage structure order:")
    for i, div in enumerate(all_divs, 1):
        classes = div.get('class', [])
        print(f"{i}. {classes}")
        
    # Check if title page has content
    if title_page:
        print(f"\nTitle page HTML:")
        print(title_page.prettify()[:500] + "...")

if __name__ == "__main__":
    debug_html()

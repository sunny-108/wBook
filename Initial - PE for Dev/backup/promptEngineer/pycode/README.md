# Markdown to PDF Book Converter

This project contains scripts to convert Markdown chapter files into a single PDF book with proper navigation, table of contents, and formatting.

## Features

- Converts individual Markdown chapter files to a single PDF book
- Creates table of contents with clickable links to chapters and sections
- Adds navigation links between chapters
- Creates PDF outline/bookmarks for navigation
- Includes front matter (about_book.md and about_author.md)
- Applies consistent styling with CSS
- Proper formatting for code blocks, tables, and images
- Adds page numbers

## Requirements

- Python 3.6+
- Required Python packages:
  - markdown
  - PyPDF2
  - weasyprint

## Installation

```bash
pip install markdown PyPDF2 weasyprint
```

## Usage

For the best results with working navigation links and page numbers:

```bash
python convert_book_with_page_numbers.py
```

This will:
1. Look for Markdown files matching `chapter*.md` in the current directory
2. Parse the TOC structure from `chapterFlow.md`
3. Create an initial HTML file with page number placeholders
4. Convert this to a temporary PDF to calculate page positions
5. Create a final HTML with accurate page numbers
6. Convert to the final PDF with working navigation and page numbers
7. Save the result as `output/book_with_page_numbers.pdf`

For books where ensuring each chapter starts on a new page is critical:

```bash
python convert_book_with_proper_page_breaks.py
```

This script:
1. Uses stronger CSS page break controls to guarantee chapters start on new pages
2. Wraps each chapter in a `chapter-container` div with explicit page break styling
3. Combines all content into a single HTML file before PDF conversion
4. Saves the result as `output/book_with_page_breaks.pdf`

Alternatively, you can use:

```bash
python convert_book_working.py
```

This will create a PDF with working navigation links but without page numbers in the TOC.

## Available Scripts

- `convert_book_with_page_numbers.py` - **Recommended**: Creates a PDF with working navigation links and page numbers in the TOC
- `convert_book_with_proper_page_breaks.py` - Creates a PDF ensuring each chapter starts on a new page
- `convert_book_working.py` - Creates a PDF with working navigation links (without TOC page numbers)
- `convert_book_final.py` - Another approach that creates a well-formatted PDF
- `convert_book_single_html.py` - Creates a PDF from a single HTML file
- `convert_book_with_navigation.py` - Uses a different approach with separate PDFs that are merged
- `convert_book_enhanced.py` - Enhanced version with better formatting
- `convert_book_improved.py` - Improved version with better TOC
- `convert_book_simple.py` - Simple version without advanced features
- `convert_book.py` - Original basic version

## File Structure

Your book directory should have:
- chapter1.md, chapter2.md, etc. (main content)
- chapterFlow.md (outline/TOC structure)
- about_book.md (optional)
- about_author.md (optional)

## Output

The final PDF will be saved in the `output` directory with:
- Clickable table of contents
- Chapter navigation links (prev/next)
- PDF bookmarks/outline for easy navigation
- Page numbers
- Consistent formatting for code blocks, tables, and images

## Notes on Navigation in PDFs

Creating PDFs with internal navigation links can be challenging. We've solved the navigation issues by:

1. Creating a single HTML file with proper anchor IDs
2. Converting that HTML directly to PDF (rather than merging separate PDFs)
3. Ensuring ID formats match between the TOC links and the heading IDs

Some minor TOC issues may persist if the sections in chapterFlow.md don't exactly match the headings in the individual chapter files.

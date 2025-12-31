# HTML to PDF Converter

This directory contains scripts to convert the `ebook.html` file to a properly formatted PDF document.

## Quick Start

### Option 1: Use the automated script (Recommended)
```bash
./convert.sh
```

This script will:
- Create a Python virtual environment
- Install all required dependencies
- Convert the HTML to PDF
- Save the output to the `output/` directory

### Option 2: Manual setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install weasyprint

# Run conversion
python convert_to_pdf.py
```

## Files

- `convert_to_pdf.py` - Main conversion script
- `convert.sh` - Automated setup and conversion script
- `requirements.txt` - Python dependencies
- `htmlFiles/ebook.html` - Source HTML file
- `output/ebook.pdf` - Generated PDF (after conversion)

## Features

The conversion script includes:
- ✅ Proper page formatting for print
- ✅ CSS styling preservation
- ✅ Page numbers
- ✅ Proper page breaks for chapters
- ✅ Code syntax highlighting
- ✅ Table formatting
- ✅ Image optimization
- ✅ Font embedding

## Requirements

- Python 3.6+
- macOS, Linux, or Windows with WSL
- System dependencies for WeasyPrint (automatically handled on most systems)

## Output

The generated PDF will be saved as `output/ebook.pdf` and includes:
- Cover page with gradient styling
- Table of contents with page links
- All chapters with proper formatting
- Professional typography and layout

## Troubleshooting

If you encounter issues:

1. **Virtual environment not activating**: Make sure you're using bash/zsh shell
2. **WeasyPrint installation fails**: You may need system dependencies:
   - macOS: `brew install cairo pango gdk-pixbuf libffi`
   - Ubuntu: `sudo apt-get install python3-dev python3-cffi libpango-1.0-0`
3. **Permission errors**: Run `chmod +x convert.sh` to make the script executable

## File Size

The generated PDF is approximately 1MB and contains the full book content with professional formatting suitable for printing or digital distribution.

#!/bin/bash
# Script to convert ebook.html to PDF with fixed page numbering
# Uses convert_to_pdf_fixed.py for proper page numbering starting from Chapter 1

echo "🔄 Converting eBook HTML to PDF with fixed page numbering..."

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🏗️  Creating virtual environment..."
    python3 -m venv .venv
fi

# Define Python executable path
PYTHON_EXE="$SCRIPT_DIR/.venv/bin/python"

# Install dependencies if needed (check if packages are already installed)
echo "📦 Installing dependencies..."
if ! $PYTHON_EXE -c "import weasyprint, PyPDF2, reportlab" 2>/dev/null; then
    $PYTHON_EXE -m pip install -r requirements.txt
else
    echo "✅ Dependencies already installed"
fi

# Run the fixed conversion
echo "🚀 Starting conversion with fixed page numbering..."
$PYTHON_EXE convert_to_pdf_fixed.py

# Verify the page numbering
echo "🔍 Verifying page numbering..."
$PYTHON_EXE verify_page_numbers.py

echo "✅ Done! Check the output directory for your PDF."
echo "📖 Chapter 1 now starts at Page 1 in the PDF!"

#!/bin/bash

# HTML to DOCX Converter Script
# Converts ebook.html to DOCX format with CSS preservation

echo "🔄 HTML to DOCX Converter"
echo "========================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install/upgrade dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Run the conversion
echo "🔄 Converting HTML to DOCX..."
python convert_to_docx.py

if [ $? -eq 0 ]; then
    echo "✅ Conversion completed successfully!"
    echo "📄 Output file: output/ebook.docx"
    echo "📖 You can open this file with Microsoft Word, LibreOffice Writer, or Google Docs"
else
    echo "❌ Conversion failed"
    exit 1
fi

# Deactivate virtual environment
deactivate

echo "🎉 Done!"

#!/bin/bash
# Simple script to convert ebook.html to PDF

echo "🔄 Converting eBook HTML to PDF..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🏗️  Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Run the conversion
echo "🚀 Starting conversion..."
python convert_to_pdf.py

echo "✅ Done! Check the output directory for your PDF."

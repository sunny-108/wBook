#!/bin/bash

# Simple HTML to DOCX Converter using Pandoc
# Best method for preserving CSS styles

echo "🔄 HTML to DOCX Converter (Pandoc)"
echo "=================================="

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "❌ Pandoc is not installed"
    echo "📦 Installing pandoc..."
    
    # Detect OS and install pandoc
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install pandoc
        else
            echo "❌ Homebrew not found. Please install pandoc manually:"
            echo "💡 https://pandoc.org/installing.html"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt update && sudo apt install pandoc
    else
        echo "❌ Please install pandoc manually:"
        echo "💡 https://pandoc.org/installing.html"
        exit 1
    fi
fi

# Check if HTML file exists
if [ ! -f "htmlFiles/ebook.html" ]; then
    echo "❌ HTML file not found: htmlFiles/ebook.html"
    exit 1
fi

# Create output directory
mkdir -p output

# Convert HTML to DOCX
echo "🔄 Converting HTML to DOCX..."
pandoc htmlFiles/ebook.html \
    -f html \
    -t docx \
    -o output/ebook.docx \
    --standalone \
    --wrap=auto \
    --metadata title="Prompt Engineering for Developers" \
    --metadata author="Sunny Shivam" \
    --metadata subject="Crafting Intelligent LLM Solutions"

if [ $? -eq 0 ]; then
    echo "✅ Successfully converted to DOCX!"
    echo "📄 Output file: output/ebook.docx"
    
    # Get file size
    size=$(ls -lh output/ebook.docx | awk '{print $5}')
    echo "📄 File size: $size"
    
    echo "🎉 Conversion complete!"
    echo "📖 You can open the DOCX file with Microsoft Word, LibreOffice Writer, or Google Docs"
    echo "💡 Pandoc preserves most CSS styles automatically"
else
    echo "❌ Conversion failed"
    exit 1
fi

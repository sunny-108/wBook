#!/bin/bash

# HTML to DOCX Converter Script - Multiple Methods
# Converts ebook.html to DOCX format with CSS preservation

echo "🔄 HTML to DOCX Converter"
echo "========================="
echo "Choose conversion method:"
echo "1. Python-based converter (custom parser)"
echo "2. Pandoc-based converter (recommended for CSS preservation)"
echo "3. Both methods"
echo ""

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo "🔧 Using Python-based converter..."
        ./convert-to-docx.sh
        ;;
    2)
        echo "🔧 Using Pandoc-based converter..."
        python3 convert_to_docx_pandoc.py
        ;;
    3)
        echo "🔧 Running both converters..."
        echo ""
        echo "▶️  Running Python-based converter..."
        ./convert-to-docx.sh
        echo ""
        echo "▶️  Running Pandoc-based converter..."
        python3 convert_to_docx_pandoc.py
        
        # Rename outputs to avoid conflicts
        if [ -f "output/ebook.docx" ]; then
            mv "output/ebook.docx" "output/ebook_pandoc.docx"
            echo "📄 Pandoc output: output/ebook_pandoc.docx"
        fi
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "🎉 Conversion complete!"
echo "📖 You can open the DOCX file(s) with Microsoft Word, LibreOffice Writer, or Google Docs"

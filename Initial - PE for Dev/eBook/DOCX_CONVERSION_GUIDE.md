# HTML to DOCX Conversion Guide

This directory contains multiple methods to convert `ebook.html` to DOCX format while preserving CSS styling.

## 🚀 Quick Start (Recommended)

### Method 1: Simple Pandoc Converter (Best Results)
```bash
# One-command conversion with excellent CSS preservation
./simple-html-to-docx.sh
```

This method:
- ✅ Preserves most CSS styles automatically
- ✅ Handles complex HTML structures
- ✅ Fast and reliable
- ✅ Creates professional-looking DOCX files
- ✅ Maintains formatting for headings, code blocks, tables, and lists

## 🔧 Alternative Methods

### Method 2: Python-based Custom Parser
```bash
# More control over the conversion process
./convert-to-docx.sh
```

This method:
- ✅ Custom style mapping
- ✅ Detailed control over formatting
- ✅ Works without pandoc
- ⚠️ May require more setup (Python packages)

### Method 3: Interactive Converter
```bash
# Choose between multiple methods
./html-to-docx.sh
```

This script lets you:
- Choose between different conversion methods
- Run multiple converters and compare results
- Get both outputs for comparison

## 📋 Prerequisites

### For Pandoc Method (Recommended)
- **macOS**: `brew install pandoc`
- **Linux**: `sudo apt install pandoc`
- **Windows**: Download from https://pandoc.org/installing.html

### For Python Method
- Python 3.6+
- Virtual environment (automatically created)
- Required packages (automatically installed):
  - `python-docx`
  - `beautifulsoup4`
  - `cssutils`
  - `lxml`

## 📄 Output Files

After conversion, you'll find:
- `output/ebook.docx` - Main DOCX file
- File size: ~113KB (compact and efficient)

## 🎨 CSS Preservation

The converters preserve:
- ✅ **Heading styles** (H1, H2, H3 with colors and fonts)
- ✅ **Code blocks** (monospace font, background colors)
- ✅ **Inline code** (monospace formatting)
- ✅ **Tables** (borders, headers, formatting)
- ✅ **Lists** (bullets, numbering, indentation)
- ✅ **Text formatting** (bold, italic, colors)
- ✅ **Paragraph spacing** (margins, line height)
- ✅ **Font families** (Arial, Courier New, etc.)

## 🔍 Comparison of Methods

| Feature | Pandoc | Python Parser |
|---------|---------|---------------|
| CSS Preservation | Excellent | Good |
| Setup Complexity | Simple | Moderate |
| Speed | Fast | Moderate |
| Customization | Limited | High |
| Reliability | Very High | High |
| File Size | Compact | Compact |

## 🛠️ Troubleshooting

### Pandoc not found
```bash
# macOS
brew install pandoc

# Linux
sudo apt update && sudo apt install pandoc
```

### Python packages not installing
```bash
# Create fresh virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Permission errors
```bash
# Make scripts executable
chmod +x *.sh
```

## 📖 Opening the DOCX File

The generated DOCX file can be opened with:
- **Microsoft Word** (Windows/Mac)
- **LibreOffice Writer** (Free, cross-platform)
- **Google Docs** (Web-based)
- **Apple Pages** (Mac)
- **WPS Office** (Cross-platform)

## 🎯 Best Practices

1. **Use the pandoc method** for best CSS preservation
2. **Test both methods** if you need specific formatting
3. **Keep the original HTML** as a backup
4. **Check the output** in your preferred word processor
5. **Customize styles** in the DOCX file if needed

## 📝 Notes

- The DOCX format has limitations compared to HTML/CSS
- Some advanced CSS features may not translate perfectly
- Page breaks and layout may differ from PDF output
- Interactive elements (links) are preserved when possible

## 🔄 Workflow Integration

You can integrate this into your workflow:

```bash
# Convert HTML to both PDF and DOCX
./convert-fixed.sh        # Creates PDF
./simple-html-to-docx.sh  # Creates DOCX
```

This gives you both formats for different use cases:
- **PDF**: For printing and fixed layout
- **DOCX**: For editing and collaboration

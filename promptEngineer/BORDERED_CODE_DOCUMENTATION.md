# Markdown to Word Converter with Bordered Code Boxes

## Overview

This enhanced Markdown to Word converter creates professional-looking documents with **bordered code boxes** and **light gray backgrounds** for both code blocks and inline code.

## Key Features

### ✅ **Bordered Code Blocks**
- **Border**: Full table border around code blocks
- **Background**: Light gray background (#F5F5F5)
- **Padding**: 0.1 inch padding on all sides
- **Font**: Consolas 9pt for optimal readability
- **Line Handling**: Intelligent line wrapping to prevent overflow

### ✅ **Inline Code Formatting**
- **Background**: Light gray background (#F0F0F0)
- **Font**: Consolas 10pt
- **Color**: Black text for readability

### ✅ **Language Support**
- Detects and displays programming language labels
- Supports syntax highlighting indicators
- Handles multiple languages (Python, JavaScript, Bash, etc.)

### ✅ **Professional Styling**
- Proper heading hierarchy with blue color scheme
- Consistent spacing and margins
- Line spacing optimization for readability
- Bold and italic text support

## Usage

### Basic Usage
```python
from md_to_docx_converter import MarkdownToDocxConverter

converter = MarkdownToDocxConverter()
converter.convert_file('input.md', 'output.docx')
```

### Processing Custom Content
```python
converter = MarkdownToDocxConverter()
converter.process_markdown_content(markdown_text)
converter.doc.save('output.docx')
```

## Code Block Examples

### Input Markdown:
```markdown
```python
def hello_world():
    print("Hello, World!")
    return "Success"
```

### Output Features:
- ✅ **Bordered table** containing the code
- ✅ **Light gray background** (#F5F5F5)
- ✅ **Monospace font** (Consolas)
- ✅ **Proper padding** (0.1 inch on all sides)
- ✅ **Language label** [PYTHON] above the code block
- ✅ **Line wrapping** for long lines

## Inline Code Examples

### Input Markdown:
```markdown
This paragraph contains `inline code` and regular text.
```

### Output Features:
- ✅ **Light gray background** (#F0F0F0) for inline code
- ✅ **Monospace font** (Consolas 10pt)
- ✅ **Seamless integration** with regular text

## Technical Implementation

### Bordered Code Boxes
- Uses Word tables with single cells to create borders
- Applies `w:shd` XML element for background color
- Sets `w:tcMar` for cell margins (padding)
- Uses `Table Grid` style for consistent borders

### Code Processing
- Intelligent line splitting at 75 characters
- Preserves indentation and code structure
- Handles long lines with proper continuation indentation

### XML Formatting
```python
# Background color implementation
shading = OxmlElement('w:shd')
shading.set(qn('w:val'), 'clear')
shading.set(qn('w:color'), 'auto')
shading.set(qn('w:fill'), 'F5F5F5')  # Light gray
```

## File Structure

```
md_to_docx_converter.py     # Main converter class
test_bordered_code.py       # Test script
chapter3.docx              # Sample output
test_bordered_code.docx     # Test output
```

## Requirements

- `python-docx` library
- Python 3.6+

## Installation

```bash
pip install python-docx
```

## Benefits

1. **Professional Appearance**: Bordered code boxes look more professional than plain text
2. **Improved Readability**: Light gray background separates code from text
3. **No Overflow**: Intelligent line wrapping prevents code from extending beyond page margins
4. **Consistency**: Uniform styling across all code blocks
5. **Language Awareness**: Displays programming language labels

## Color Scheme

- **Code Block Background**: #F5F5F5 (Light gray)
- **Inline Code Background**: #F0F0F0 (Very light gray)
- **Headings**: #2E74B5 (Professional blue)
- **Code Text**: #000000 (Black for readability)

## Example Output

The converter transforms this markdown:

```markdown
# Chapter Title

Here's a code example:

```python
def calculate_sum(a, b):
    return a + b

result = calculate_sum(5, 3)
print(f"Result: {result}")
```

This uses `inline code` in text.
```

Into a professional Word document with:
- Bordered code block with light gray background
- Language label [PYTHON]
- Inline code with light gray background
- Professional typography and spacing

## Success Indicators

✅ **Bordered Code Blocks**: Full table borders around code  
✅ **Light Gray Background**: Consistent background color  
✅ **Proper Padding**: 0.1 inch padding on all sides  
✅ **Line Wrapping**: No code overflow issues  
✅ **Language Labels**: Programming language indicators  
✅ **Inline Code**: Background highlighting for inline code  
✅ **Professional Styling**: Clean, readable formatting  

Your converter now creates professional-looking documents with bordered code boxes and light gray backgrounds exactly as requested!

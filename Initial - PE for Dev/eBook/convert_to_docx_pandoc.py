#!/usr/bin/env python3
"""
Alternative HTML to DOCX Converter using Pandoc
This approach uses pandoc which is excellent at preserving CSS styles
"""

import subprocess
import sys
from pathlib import Path
import shutil

def check_pandoc():
    """Check if pandoc is installed"""
    return shutil.which('pandoc') is not None

def install_pandoc():
    """Install pandoc using system package manager"""
    import platform
    
    system = platform.system().lower()
    
    if system == 'darwin':  # macOS
        print("📦 Installing pandoc via Homebrew...")
        try:
            subprocess.run(['brew', 'install', 'pandoc'], check=True)
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install pandoc via Homebrew")
            print("💡 Please install Homebrew first: https://brew.sh")
            print("💡 Or install pandoc manually: https://pandoc.org/installing.html")
            return False
    elif system == 'linux':
        print("📦 Installing pandoc via apt...")
        try:
            subprocess.run(['sudo', 'apt', 'update'], check=True)
            subprocess.run(['sudo', 'apt', 'install', 'pandoc'], check=True)
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install pandoc via apt")
            print("💡 Please install pandoc manually: https://pandoc.org/installing.html")
            return False
    else:
        print("❌ Automatic pandoc installation not supported on this platform")
        print("💡 Please install pandoc manually: https://pandoc.org/installing.html")
        return False

def convert_html_to_docx_pandoc(html_file, output_file):
    """Convert HTML to DOCX using pandoc"""
    
    # Pandoc command with options to preserve styling
    cmd = [
        'pandoc',
        str(html_file),
        '-f', 'html',
        '-t', 'docx',
        '-o', str(output_file),
        '--standalone',
        '--wrap=auto',
        '--reference-doc=reference.docx'  # Optional: use a reference document for styling
    ]
    
    # Remove reference doc option if file doesn't exist
    if not Path('reference.docx').exists():
        cmd = cmd[:-2]  # Remove last two arguments
    
    try:
        print(f"🔄 Running pandoc conversion...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully converted to DOCX: {output_file}")
            return True
        else:
            print(f"❌ Pandoc conversion failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Pandoc conversion failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def create_reference_docx():
    """Create a reference DOCX file with custom styling"""
    try:
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
        from docx.enum.style import WD_STYLE_TYPE
        
        doc = Document()
        
        # Customize styles
        # Heading 1
        h1_style = doc.styles['Heading 1']
        h1_style.font.size = Pt(20)
        h1_style.font.color.rgb = RGBColor(44, 62, 80)  # #2c3e50
        h1_style.font.bold = True
        
        # Heading 2
        h2_style = doc.styles['Heading 2']
        h2_style.font.size = Pt(16)
        h2_style.font.color.rgb = RGBColor(52, 152, 219)  # #3498db
        h2_style.font.bold = True
        
        # Heading 3
        h3_style = doc.styles['Heading 3']
        h3_style.font.size = Pt(14)
        h3_style.font.color.rgb = RGBColor(41, 128, 185)  # #2980b9
        h3_style.font.bold = True
        
        # Normal text
        normal_style = doc.styles['Normal']
        normal_style.font.size = Pt(10.5)
        normal_style.font.name = 'Arial'
        
        # Create code style
        try:
            code_style = doc.styles.add_style('Code', WD_STYLE_TYPE.CHARACTER)
            code_style.font.name = 'Courier New'
            code_style.font.size = Pt(9)
        except:
            pass
        
        # Save reference document
        doc.save('reference.docx')
        print("✅ Created reference.docx for styling")
        return True
        
    except ImportError:
        print("⚠️  python-docx not available, skipping reference document creation")
        return False
    except Exception as e:
        print(f"⚠️  Failed to create reference document: {e}")
        return False

def main():
    """Main function"""
    print("🔄 HTML to DOCX Converter (Pandoc Method)")
    print("=" * 45)
    
    # Define paths
    script_dir = Path(__file__).parent
    html_file = script_dir / "htmlFiles" / "ebook.html"
    output_file = script_dir / "output" / "ebook.docx"
    
    # Check if HTML file exists
    if not html_file.exists():
        print(f"❌ HTML file not found: {html_file}")
        return
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if pandoc is installed
    if not check_pandoc():
        print("❌ Pandoc is not installed")
        print("🤔 Would you like to install pandoc? (y/n)")
        
        # For automation, we'll try to install
        if install_pandoc():
            print("✅ Pandoc installed successfully")
        else:
            print("❌ Failed to install pandoc")
            return
    else:
        print("✅ Pandoc is available")
    
    # Create reference document (optional)
    create_reference_docx()
    
    # Convert
    success = convert_html_to_docx_pandoc(html_file, output_file)
    
    if success:
        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"📄 File size: {file_size:.2f} MB")
        print(f"🎉 Conversion complete!")
        print(f"📖 Open the DOCX file with Microsoft Word or LibreOffice Writer")
        print(f"💡 Pandoc generally preserves CSS styles better than custom parsers")
    else:
        print("❌ Conversion failed")

if __name__ == "__main__":
    main()

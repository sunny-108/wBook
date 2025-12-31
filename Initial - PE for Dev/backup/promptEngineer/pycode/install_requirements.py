#!/usr/bin/env python
"""
Simple script to install required packages for adding page numbers to PDFs.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Install required packages."""
    print("Installing required packages for PDF page numbering...")
    
    packages = [
        "PyPDF2",
        "weasyprint", 
        "markdown",
        "reportlab"  # Optional, for the advanced script
    ]
    
    success_count = 0
    for package in packages:
        print(f"\nInstalling {package}...")
        if install_package(package):
            success_count += 1
    
    print(f"\n🎉 Installation complete! {success_count}/{len(packages)} packages installed successfully.")
    
    if success_count >= 3:  # PyPDF2, weasyprint, markdown are the minimum required
        print("\n✅ You can now run:")
        print("   python add_chapter_page_numbers.py")
    else:
        print("\n❌ Some required packages failed to install. Please install them manually:")
        print("   pip install PyPDF2 weasyprint markdown")

if __name__ == "__main__":
    main()

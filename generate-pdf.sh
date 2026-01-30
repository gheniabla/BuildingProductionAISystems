#!/bin/bash

# PDF Generation Script for "Building Production AI Systems"
#
# This script generates a professional PDF from the course markdown files.
#
# Prerequisites:
# - pandoc: brew install pandoc
# - LaTeX: brew install --cask mactex (or brew install basictex)
# - Or use the HTML method which requires no additional installs

OUTPUT_DIR="/Users/abla/Desktop/2026-Jan/Codes/BuildingProductionAISystems"
COMBINED_FILE="$OUTPUT_DIR/Building-Production-AI-Systems-Complete.md"

echo "======================================"
echo "Building Production AI Systems"
echo "PDF Generation"
echo "======================================"

# Method 1: Using Pandoc with LaTeX (Best quality)
generate_with_pandoc() {
    echo ""
    echo "Method 1: Generating PDF with Pandoc + LaTeX..."

    if ! command -v pandoc &> /dev/null; then
        echo "Pandoc not found. Install with: brew install pandoc"
        return 1
    fi

    pandoc "$COMBINED_FILE" \
        -o "$OUTPUT_DIR/Building-Production-AI-Systems.pdf" \
        --from markdown \
        --pdf-engine=xelatex \
        --toc \
        --toc-depth=3 \
        --highlight-style=tango \
        -V geometry:margin=1in \
        -V fontsize=11pt \
        -V documentclass=report \
        -V colorlinks=true \
        -V linkcolor=blue \
        -V urlcolor=blue \
        -V toccolor=gray \
        --metadata title="Building Production AI Systems" \
        --metadata author="Course Development Team" \
        --metadata date="$(date +%Y-%m-%d)"

    if [ $? -eq 0 ]; then
        echo "✓ PDF generated: $OUTPUT_DIR/Building-Production-AI-Systems.pdf"
        return 0
    else
        echo "✗ Pandoc PDF generation failed"
        return 1
    fi
}

# Method 2: Using Pandoc to HTML then print
generate_html() {
    echo ""
    echo "Method 2: Generating HTML (for browser printing)..."

    if ! command -v pandoc &> /dev/null; then
        echo "Pandoc not found. Install with: brew install pandoc"
        return 1
    fi

    pandoc "$COMBINED_FILE" \
        -o "$OUTPUT_DIR/Building-Production-AI-Systems.html" \
        --from markdown \
        --to html5 \
        --standalone \
        --toc \
        --toc-depth=3 \
        --highlight-style=tango \
        --metadata title="Building Production AI Systems" \
        --css="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css" \
        -V mainfont="Helvetica" \
        -V monofont="Monaco"

    if [ $? -eq 0 ]; then
        echo "✓ HTML generated: $OUTPUT_DIR/Building-Production-AI-Systems.html"
        echo "  Open in browser and use Print → Save as PDF"
        return 0
    else
        echo "✗ HTML generation failed"
        return 1
    fi
}

# Method 3: Using grip (GitHub-style rendering)
generate_with_grip() {
    echo ""
    echo "Method 3: Using grip for GitHub-style rendering..."

    if ! command -v grip &> /dev/null; then
        echo "grip not found. Install with: pip install grip"
        return 1
    fi

    echo "Starting grip server at http://localhost:6419"
    echo "Open in browser, then use Print → Save as PDF"
    echo "Press Ctrl+C to stop the server"

    grip "$COMBINED_FILE" --browser
}

# Check for combined file
if [ ! -f "$COMBINED_FILE" ]; then
    echo "Error: Combined markdown file not found at $COMBINED_FILE"
    exit 1
fi

echo ""
echo "File to convert: $COMBINED_FILE"
echo ""

# Try methods in order of preference
if generate_with_pandoc; then
    echo ""
    echo "======================================"
    echo "PDF generation complete!"
    echo "Output: $OUTPUT_DIR/Building-Production-AI-Systems.pdf"
    echo "======================================"
    exit 0
fi

echo ""
echo "Falling back to HTML method..."
if generate_html; then
    echo ""
    echo "======================================"
    echo "HTML generation complete!"
    echo ""
    echo "To create PDF:"
    echo "1. Open: $OUTPUT_DIR/Building-Production-AI-Systems.html"
    echo "2. In browser: File → Print → Save as PDF"
    echo "======================================"
    exit 0
fi

echo ""
echo "======================================"
echo "ALTERNATIVE METHODS:"
echo ""
echo "1. VS Code: Install 'Markdown PDF' extension"
echo "   Right-click .md file → 'Markdown PDF: Export (pdf)'"
echo ""
echo "2. Online: Use https://md2pdf.netlify.app/"
echo "   Paste the markdown content and download PDF"
echo ""
echo "3. Typora: Open .md file and export to PDF"
echo ""
echo "======================================"

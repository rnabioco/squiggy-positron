# Squiggy Manuscript

Academic manuscript describing the Squiggy Positron extension for interactive nanopore signal visualization.

## Files

- **`squiggy-manuscript.typ`** - Main manuscript (Typst format)
- **`oup-template.typ`** - Oxford University Press journal template
- **`references.bib`** - BibTeX bibliography with all citations

## Prerequisites

Install Typst on your system:

**macOS/Linux (Homebrew):**
```bash
brew install typst
```

**Other methods:**
- Cargo: `cargo install --git https://github.com/typst/typst --locked typst-cli`
- Download binary: https://github.com/typst/typst/releases
- Arch Linux: `pacman -S typst`
- Windows: `winget install --id Typst.Typst`

## Compiling to PDF

**One-time compilation:**
```bash
typst compile squiggy-manuscript.typ
```

This creates `squiggy-manuscript.pdf` in the current directory.

**Watch mode (auto-recompile on save):**
```bash
typst watch squiggy-manuscript.typ
```

**Custom output name:**
```bash
typst compile squiggy-manuscript.typ output-name.pdf
```

## Customization

### Change Font
If "New Computer Modern" is not available, edit `squiggy-manuscript.typ`:
```typst
set text(
  font: "Linux Libertine",  // or another font
  size: 10pt
)
```

### Change Citation Style
The manuscript uses Nature style. To change it, edit the bibliography line:
```typst
#bibliography("references.bib", title: "References", style: "ieee")
```

Available styles: `"nature"`, `"ieee"`, `"apa"`, `"chicago-notes"`, `"mla"`, etc.

## Adding References

Edit `references.bib` and add new entries in BibTeX format. Then cite in the manuscript using:
```typst
@CitationKey
```

The bibliography will automatically update.

## Output

The compiled PDF will be saved as `squiggy-manuscript.pdf` and includes:
- Formatted title page with authors and affiliations
- Abstract and keywords
- Main manuscript sections
- Comparison table with existing tools
- Automatically formatted references

## Generating Figures

### Architecture Diagram (Figure 1)

The architecture diagram uses Mermaid. Generate it before compiling:

**Install Mermaid CLI:**
```bash
npm install -g @mermaid-js/mermaid-cli
```

**Generate SVG:**
```bash
mmdc -i architecture-diagram.mmd -o architecture-diagram.svg
```

**Or use the online tool:**
1. Visit https://mermaid.live/
2. Paste contents of `architecture-diagram.mmd`
3. Download as SVG and save to this directory

**Update the manuscript:**
Once you have `architecture-diagram.svg`, uncomment this line in `squiggy-manuscript.typ`:
```typst
image("architecture-diagram.svg", width: 100%)
```

### Figure 2 (UI Screenshot)

Replace the placeholder with an actual screenshot of Squiggy in Positron IDE.

## Quick Start

```bash
# Compile and open (macOS)
typst compile squiggy-manuscript.typ && open squiggy-manuscript.pdf

# Compile and open (Linux)
typst compile squiggy-manuscript.typ && xdg-open squiggy-manuscript.pdf
```

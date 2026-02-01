# Course Slides — Building Production AI Systems

Slide decks built with [Marp](https://marp.app/) (Markdown Presentation Ecosystem).

## Prerequisites

- Node.js 18+
- npm

## Setup

```bash
cd slides
npm install
```

## Usage

### Live Preview (recommended for presenting)

```bash
npm run serve
```

Opens a live-reloading server. Edit the markdown files and see changes instantly.

### Export to PDF

```bash
npm run build
```

PDFs are written to `slides/dist/`.

### Export to HTML

```bash
npm run export
```

HTML files are written to `slides/dist/`.

## Structure

```
slides/
├── package.json              # marp-cli dependency
├── .marprc.yml               # Marp config
├── theme/
│   └── course-theme.css      # Custom CSS theme
├── week1/
│   ├── ch1-production-landscape.md
│   └── ch2-genai-review.md
└── dist/                     # Build output (git-ignored)
```

## Mermaid Diagrams

Marp CLI renders Mermaid diagrams via its built-in Chromium engine. Diagrams render automatically in `serve` mode and in PDF/HTML exports.

## Customizing the Theme

Edit `theme/course-theme.css`. The theme uses an indigo/blue palette that matches the Mermaid diagram color scheme used in the course notes.

Key CSS classes:
- `lead` — Title slides (centered, gradient background)
- `two-col` — Two-column grid layout
- `diagram` — Slides with large diagrams (extra padding)

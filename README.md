# Portfolio Project

Data portfolio of **Nikolas Scevko** — a hand-built dark "data-editorial" theme (no remote theme, no framework).

Live at: https://scevinno.github.io/Portfolio_Project

## Adding a project

1. Drop a markdown file in `_posts/` named `YYYY-MM-DD-slug.md`.
2. Front matter drives the case-study card on the home page:

```yaml
---
layout: post
title: Project Title
image: "/img/posts/cover.svg"        # cover image, lives in img/posts/
tags: [Machine Learning, Python]
summary: "One-to-two sentence hook shown on the card and page header."
stack: "Python · pandas · scikit-learn"
metrics:                              # up to ~3 headline numbers
  - value: "0.76"
    label: "R² on unseen sales"
---
```

3. Write the body in plain markdown — code blocks, tables and blockquote callouts are styled automatically.

## Stack

- Jekyll (GitHub Pages native build), kramdown + rouge
- Custom layouts in `_layouts/`, one stylesheet (`assets/css/style.css`), one small JS file (`assets/js/main.js`)
- Fonts: Space Grotesk (display), Inter (body), JetBrains Mono (data labels & code)

# Webpage and GitHub Pages

This document is for website maintainers. Paths and commands below are relative to the repository root.

`webpage/dist/` is a self-contained static site. It can be previewed locally without a build step:

```bash
python3 -m http.server 8123 --directory webpage/dist
```

The included workflow at `.github/workflows/pages.yml` publishes this directory through GitHub Pages. In the repository settings, set Pages → Build and deployment → Source to **GitHub Actions**. The public repository is [2022hpsk/SpeakerMemR1](https://github.com/2022hpsk/SpeakerMemR1), and the website URL is [https://2022hpsk.github.io/SpeakerMemR1/](https://2022hpsk.github.io/SpeakerMemR1/). All Paper links point to [arXiv:2609.26780](https://arxiv.org/abs/2609.26780). Author names and affiliation are maintained in `dist/index.html`; keep the `zheng2026speakermemr1` BibTeX synchronized with the top-level README.

The website uses self-contained SVG link icons. README resource badges use Shields.io. The three main figures are bundled as PDF originals and WebP previews in `webpage/dist/assets/`; all three figures also have PNG previews for the README. When updating figures, regenerate the previews as well as replacing the PDFs.

## Release updates

- The workflow listens to `master` and deploys only `webpage/dist/`.
- Keep the README arXiv badge and all website Paper links synchronized with the paper record.
- The BibTeX panel and copy button in `dist/index.html` use the citation event handlers in `dist/app.js`.
- The manuscript PDF is not bundled; Paper links use the arXiv abstract page, which provides access to the PDF.

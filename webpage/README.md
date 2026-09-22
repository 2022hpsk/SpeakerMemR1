# Webpage and GitHub Pages

This document is for website maintainers. Paths and commands below are relative to the repository root.

`webpage/dist/` is a self-contained static site. It can be previewed locally without a build step:

```bash
python3 -m http.server 8123 --directory webpage/dist
```

The included workflow at `.github/workflows/pages.yml` publishes this directory through GitHub Pages. In the repository settings, set Pages → Build and deployment → Source to **GitHub Actions**. The public repository is [2022hpsk/SpeakerMemR1](https://github.com/2022hpsk/SpeakerMemR1), and the website URL is [https://2022hpsk.github.io/SpeakerMemR1/](https://2022hpsk.github.io/SpeakerMemR1/). Paper and citation controls display **Coming soon** until the arXiv release. Add the real author names, affiliations, author homepages, and verified BibTeX when they are available; do not publish example identities or invented arXiv identifiers.

The website uses self-contained SVG link icons. README resource badges use Shields.io. The three main figures are bundled as PDF originals and WebP previews in `webpage/dist/assets/`; all three figures also have PNG previews for the README. When updating figures, regenerate the previews as well as replacing the PDFs.

## Release updates

- The workflow listens to `master` and deploys only `webpage/dist/`.
- Replace the README Paper badge target (`#paper`) with the real arXiv URL, and update its Coming soon label.
- Replace disabled Paper and BibTeX controls in `dist/index.html` with working links/actions when released. The citation event handlers in `dist/app.js` are optional until these controls return.
- Add verified author information to the README and website.
- The manuscript PDF is not bundled. Update the Paper controls to link to arXiv when the paper is released.

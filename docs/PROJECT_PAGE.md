# TED project page

The buildless academic project page is in `project-page/`. It uses original figures from the TED paper; source details are recorded in `PROJECT_PAGE_ASSETS.json`.

- `project-page/index.html`: text, authors, resource links and BibTeX
- `project-page/style.css`: layout and responsive typography
- `project-page/script.js`: gallery, image zoom and citation copy
- `project-page/assets/`: original paper figures in WebP format

GitHub Actions publishes only `project-page/` to GitHub Pages. No installation or build is required. The repository's Python package and documentation are not part of the deployed artifact.

The Pages workflow runs on changes to this directory or the workflow itself, and can also be run manually. In repository Settings > Pages, the source is GitHub Actions.

Public page: https://barraki72268-sketch.github.io/TED/

The public project page is now maintained at https://github.com/barraki72268-sketch/TED. The original long Pages address redirects to the short public address. The original page source remains here as a fallback.

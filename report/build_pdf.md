# Building the report PDF

The report source is `REPORT.md`; images are pulled from `../figures/` and `img/`.

Option A (pandoc + any browser):

    cd report
    pandoc REPORT.md -s --embed-resources --resource-path=. -c report.css -o report.html

then open `report.html` in Chrome/Edge and Print → Save as PDF (margins: default, background graphics: on).

Option B: paste `REPORT.md` into Word / Google Docs and insert the PNGs from `figures/`.

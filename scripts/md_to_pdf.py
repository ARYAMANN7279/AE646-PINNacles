import re
import sys
import markdown
from xhtml2pdf import pisa


def fix_pre_linebreaks(html):
    """xhtml2pdf does not reliably honor white-space:pre-wrap, so newlines inside
    <pre> blocks must be converted to explicit <br/> tags or the block renders as
    one run-on line."""
    def _sub(m):
        return m.group(0).replace("\n", "<br/>\n")
    return re.sub(r"<pre>.*?</pre>", _sub, html, flags=re.DOTALL)

CSS = """
@page {
    size: letter;
    margin: 1.6cm 1.5cm 1.8cm 1.5cm;
    @frame footer_frame {
        -pdf-frame-content: footer_content;
        bottom: 0.8cm; margin-left: 1.5cm; margin-right: 1.5cm; height: 1cm;
    }
}
body { font-family: "Helvetica", sans-serif; font-size: 10.3pt; line-height: 1.42; color: #1a1a1a; }
h1 { font-size: 18pt; margin-top: 0; margin-bottom: 6pt; color: #111; border-bottom: 1.4pt solid #333; padding-bottom: 4pt; }
h2 { font-size: 13.5pt; margin-top: 16pt; margin-bottom: 6pt; color: #16324f; border-bottom: 0.6pt solid #aaa; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin-top: 12pt; margin-bottom: 4pt; color: #16324f; }
p { margin: 5pt 0; text-align: left; }
ul, ol { margin: 4pt 0 8pt 0; padding-left: 16pt; }
li { margin: 2pt 0; }
strong { color: #000; }
hr { border: none; border-top: 0.5pt solid #bbb; margin: 10pt 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 10pt 0; }
th { background-color: #e8edf2; border: 0.5pt solid #999; padding: 3pt 4pt; font-size: 8.3pt; text-align: left; font-weight: bold; }
td { border: 0.5pt solid #999; padding: 3pt 4pt; font-size: 8.3pt; }
pre { background-color: #f4f4f4; border: 0.5pt solid #ccc; padding: 6pt 8pt; font-size: 8.5pt; font-family: "Courier New", monospace; white-space: pre-wrap; margin: 6pt 0; }
code { font-family: "Courier New", monospace; background-color: #f2f2f2; font-size: 9pt; padding: 0 2pt; }
pre code { background-color: transparent; padding: 0; }
a { color: #1155aa; }
blockquote { border-left: 2pt solid #ccc; margin: 6pt 0; padding-left: 10pt; color: #444; }
#footer_content { font-size: 8pt; color: #888; text-align: center; }
"""

def convert(md_path, pdf_path, title):
    with open(md_path, "r") as f:
        md_text = f.read()

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    html_body = fix_pre_linebreaks(html_body)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
<div id="footer_content">{title} &nbsp;|&nbsp; AE646: Scientific Machine Learning for Fluid Mechanics &nbsp;|&nbsp; Page <pdf:pagenumber/></div>
{html_body}
</body></html>"""

    with open(pdf_path, "wb") as out:
        result = pisa.CreatePDF(html, dest=out)
    if result.err:
        print(f"FAILED: {md_path} -> {pdf_path} ({result.err} errors)")
        return False
    print(f"OK: {md_path} -> {pdf_path}")
    return True

if __name__ == "__main__":
    import os
    # Resolve paths relative to the repo root (this script lives in <root>/scripts/),
    # so the script is portable and runnable from anywhere after cloning.
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DOCS = os.path.join(ROOT, "docs")
    jobs = [
        ("PROPOSAL.md", "PROPOSAL.pdf", "Stage 1: Project Proposal"),
        ("INTERIM_REPORT.md", "INTERIM_REPORT.pdf", "Stage 2: Interim Report"),
        ("FINAL_REPORT.md", "FINAL_REPORT.pdf", "Stage 3: Final Report"),
    ]
    ok = True
    for md, pdf, title in jobs:
        ok = convert(os.path.join(DOCS, md), os.path.join(DOCS, pdf), title) and ok
    sys.exit(0 if ok else 1)

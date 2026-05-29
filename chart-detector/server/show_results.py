"""
downloads/ 폴더를 스캔해서 사이트별 차트 갤러리 HTML을 생성하고 브라우저로 엽니다.
사용법: python show_results.py
"""

import os
import base64
import webbrowser
from pathlib import Path
from datetime import datetime

DOWNLOADS = Path(__file__).parent / "downloads"
OUTPUT    = Path(__file__).parent / "results.html"

EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def site_label(folder_name: str) -> str:
    return folder_name.replace("_", ".")


def encode_image(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/{mime};base64,{data}"


def collect() -> dict[str, dict[str, list[Path]]]:
    """{ site_folder: { article_folder: [image, ...] } }"""
    result = {}
    if not DOWNLOADS.exists():
        return result
    for site_dir in sorted(DOWNLOADS.iterdir()):
        if not site_dir.is_dir():
            continue
        articles = {}
        for article_dir in sorted(site_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
            if not article_dir.is_dir():
                continue
            images = sorted(
                [f for f in article_dir.iterdir() if f.suffix.lower() in EXTS],
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if images:
                articles[article_dir.name] = images
        if articles:
            result[site_dir.name] = articles
    return result


def build_html(data: dict[str, dict[str, list[Path]]]) -> str:
    total = sum(len(imgs) for articles in data.values() for imgs in articles.values())
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    site_blocks = ""
    for site_folder, articles in data.items():
        site_total = sum(len(imgs) for imgs in articles.values())
        article_blocks = ""
        for article_folder, images in articles.items():
            title = article_folder.replace("_", " ")
            cards = ""
            for img_path in images:
                src = encode_image(img_path)
                mtime = datetime.fromtimestamp(img_path.stat().st_mtime).strftime("%m-%d %H:%M")
                cards += f"""
                <div class="card">
                    <img src="{src}" alt="{img_path.name}" loading="lazy">
                    <div class="meta">{img_path.name}<br><span>{mtime}</span></div>
                </div>"""
            article_blocks += f"""
            <div class="article">
                <h3>{title} <span class="count">{len(images)}장</span></h3>
                <div class="grid">{cards}
                </div>
            </div>"""

        site_blocks += f"""
        <section>
            <h2>{site_label(site_folder)} <span class="count">{site_total}장</span></h2>
            {article_blocks}
        </section>"""

    if not site_blocks:
        site_blocks = '<p class="empty">downloads/ 폴더에 저장된 차트가 없습니다.<br>서버를 실행하고 뉴스 사이트를 탐색해 보세요.</p>'

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>차트 탐지 결과</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f4f5f7; color: #222; padding: 24px; }}
  header {{ margin-bottom: 28px; }}
  header h1 {{ font-size: 1.5rem; font-weight: 700; }}
  header p  {{ color: #666; font-size: 0.875rem; margin-top: 4px; }}
  .badge {{ display: inline-block; background: #2563eb; color: #fff;
            border-radius: 999px; padding: 2px 10px; font-size: 0.8rem;
            margin-left: 8px; vertical-align: middle; }}
  section {{ background: #fff; border-radius: 12px; padding: 20px 24px;
             margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  h2 {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 16px; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
  .article {{ margin-bottom: 20px; }}
  h3 {{ font-size: 0.95rem; font-weight: 600; color: #374151; margin-bottom: 10px;
        background: #f9fafb; padding: 6px 10px; border-radius: 6px; border-left: 3px solid #2563eb; }}
  .count {{ color: #2563eb; font-size: 0.9rem; font-weight: 500; margin-left: 6px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }}
  .card {{ border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden;
           background: #fafafa; transition: box-shadow .15s; }}
  .card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,.12); }}
  .card img {{ width: 100%; height: 150px; object-fit: cover; display: block;
               background: #e5e7eb; }}
  .meta {{ padding: 8px 10px; font-size: 0.72rem; color: #555;
           word-break: break-all; line-height: 1.4; }}
  .meta span {{ color: #999; }}
  .empty {{ text-align: center; color: #888; padding: 48px 0; line-height: 2; }}
</style>
</head>
<body>
<header>
  <h1>차트 탐지 결과 <span class="badge">총 {total}장</span></h1>
  <p>생성: {generated} &nbsp;|&nbsp; 저장 경로: downloads/</p>
</header>
{site_blocks}
</body>
</html>"""


def main():
    data = collect()
    html = build_html(data)
    OUTPUT.write_text(html, encoding="utf-8")

    total = sum(len(imgs) for articles in data.values() for imgs in articles.values())
    print(f"[결과] {len(data)}개 사이트, 총 {total}장")
    for site_folder, articles in data.items():
        site_total = sum(len(imgs) for imgs in articles.values())
        print(f"  {site_label(site_folder)}: {site_total}장 ({len(articles)}개 기사)")
        for article_folder, images in articles.items():
            print(f"    └ {article_folder[:50]}: {len(images)}장")
    print(f"\n[저장] {OUTPUT}")

    webbrowser.open(OUTPUT.as_uri())


if __name__ == "__main__":
    main()

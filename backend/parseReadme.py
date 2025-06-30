import mistune
from bs4 import BeautifulSoup
import json

def parse_html_to_json(html):
    """
    Parses provided HTML to extract:
      - main sections (<p><strong>…</strong></p>)
      - subsections (any <li>, with or without <strong>)
      - points (text after <strong> plus any nested <li>)
    
    Output format:
    [
      {
        "section": "Main Section Title",
        "subsections": [
          {
            "title": "Subsection Title" or None,
            "points": [
              "text immediately after the <strong>…</strong>",
              "nested point 1",
              "nested point 2",
              …
            ]
          },
          …
        ]
      },
      …
    ]
    """
    soup = BeautifulSoup(html, 'html.parser')
    result = []

    # 1) Main sections
    for main_p in soup.find_all('p'):
        main_strong = main_p.find('strong')
        if not main_strong:
            continue
        section_title = main_strong.get_text(strip=True)
        section = {"section": section_title, "subsections": []}

        # 2) Collect until next main section
        for sib in main_p.next_siblings:
            if getattr(sib, 'name', None) == 'p' and sib.find('strong'):
                break
            if getattr(sib, 'name', None) != 'ul':
                continue

            # 3) Each top-level LI becomes its own subsection
            for li in sib.find_all('li', recursive=False):
                # Find <strong> if present
                strong_tag = li.find('strong')
                if strong_tag:
                    title = strong_tag.get_text(strip=True)
                    # Remove the <strong> node so its text doesn't reappear
                    strong_tag.extract()
                else:
                    title = None

                # Text immediately inside this LI (after removing <strong>)
                # We exclude text from any nested <ul>
                direct_text = ''.join(
                    t.strip() 
                    for t in li.find_all(text=True, recursive=False)
                    if t.strip()
                )
                points = [direct_text] if direct_text else []

                # 4) Collect any nested <li> as additional points
                nested_ul = li.find('ul')
                if nested_ul:
                    for deep_li in nested_ul.find_all('li'):
                        txt = deep_li.get_text(strip=True)
                        if txt:
                            points.append(txt)

                section["subsections"].append({
                    "title": title,
                    "points": points
                })

        result.append(section)

    return result


def parse_readme_to_json(readme_data):
    return parse_html_to_json(mistune.html(readme_data))
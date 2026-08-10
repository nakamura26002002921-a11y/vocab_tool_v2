import argparse
import re
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


def scrape(url, max_words=100):
    html = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    ).text

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    return " ".join(text.split()[:max_words])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", required=True)
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--max-words", type=int, default=100)
    args = parser.parse_args()

    with DDGS() as ddgs:
        results = list(ddgs.text(
            query=args.word,
            region="wt-wt",
            safesearch="moderate",
            max_results=args.max_results,
        ))

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['title']}")
        print(r["href"])
        print(scrape(r["href"], args.max_words))


if __name__ == "__main__":
    main()

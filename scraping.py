"""
scraping.py
-----------
複数のロシア語辞書・翻訳サイトから単語データを取得し、SQLiteの raw_data テーブルに
キャッシュとして保存するモジュール。ソースの性質に応じて2つの取得・抽出方式を使い分ける。

【方式1: wikitext方式】 (source_type = "mediawiki_wikitext")
  ru.wiktionary.org 等、MediaWiki上の辞書サイト向け。
  action=parse&prop=wikitext で生wiki記法を取得し、正規表現でセクション・テンプレート
  （{{семантика|...}} {{пример|...}} 等）を解釈する。HTMLレンダリング結果よりテンプレート
  構文の方が変更されにくいため、この方式の方が壊れにくい。

【方式2: HTML方式】 (source_type = "html", デフォルト)
  gufo.me・slovaronline・multitran・reverso等、MediaWiki以外の一般的な辞書/翻訳サイト向け。
  通常のHTMLページを取得し、config.json の field_map に列挙された正規表現パターンを
  順に適用してテキストを抽出する（lxml/XPathではなく、CSSクラス名やタグ構造の変化に
  多少強い緩めの正規表現ベースの抽出とし、複数パターンをフォールバックとして試せるようにした）。
  各サイトはHTML構造の変更が比較的頻繁なため、正確性は都度の実行結果を見て調整が必要。

- キャッシュキー: (word, source_url)
- 既にDBに該当レコードがあればHTTPリクエストを行わずスキップする。
- 1単語につき config.json の scraping.sources に列挙された全ソースを試行し、
  ソースごとの抽出結果をまとめた dict を返す。
- 下流（summarize.py）との互換性のため、戻り値の形は常に
  {"status": ..., "extracted": {field_name: [text, ...]}} を維持する。
"""

from __future__ import annotations

import json
import re
import time
from typing import Optional

import requests

from common import get_connection, now_iso, setup_logger

logger = setup_logger("logs/errors.log")


class ScrapingError(Exception):
    pass


# ---------------------------------------------------------------------------
# 共通ユーティリティ
# ---------------------------------------------------------------------------
def _build_url(url_template: str, word: str) -> str:
    """表示用/キャッシュキー用/取得先のページURLを組み立てる。"""
    return url_template.format(word=word)


def _build_api_url(api_url_template: str) -> str:
    """config.json の api_url_template（例: "https://ru.wiktionary.org/w/api.php"）
    をそのまま返す。末尾の / は落とす。"""
    return api_url_template.rstrip("/")


def _fetch_html_page(url: str, user_agent: str, timeout: int, max_retries: int) -> str:
    """通常のHTMLページを取得する（HTML方式ソース用）。"""
    headers = {
        "User-Agent": user_agent,
        "Accept-Language": "ru,en;q=0.8",
    }
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 404:
                raise ScrapingError(f"404 Not Found: {url}")
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except ScrapingError:
            raise
        except Exception as e:  # noqa: BLE001
            last_exc = e
            logger.warning(
                "scraping: fetch failed (attempt %d/%d) url=%s error=%s",
                attempt, max_retries, url, e,
            )
            if attempt < max_retries:
                time.sleep(1.5 * attempt)

    raise ScrapingError(f"Failed to fetch {url}: {last_exc}")


def _fetch_wikitext(api_url: str, page: str, user_agent: str, timeout: int, max_retries: int) -> str:
    """action=parse&prop=wikitext を叩いてページの生wikitextを返す。"""
    headers = {"User-Agent": user_agent}
    params = {
        "action": "parse",
        "page": page,
        "prop": "wikitext",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
    }
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(api_url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                err = data["error"]
                code = err.get("code", "")
                info = err.get("info", code)
                if code == "missingtitle":
                    raise ScrapingError(f"404 Not Found (missingtitle): {info}")
                raise ScrapingError(f"MediaWiki API error ({code}): {info}")

            wikitext = data.get("parse", {}).get("wikitext")
            if wikitext is None:
                raise ScrapingError(f"Unexpected API response shape (no parse.wikitext): {api_url}?page={page}")
            return wikitext
        except ScrapingError:
            raise
        except Exception as e:  # noqa: BLE001
            last_exc = e
            logger.warning(
                "scraping: fetch failed (attempt %d/%d) api_url=%s page=%s error=%s",
                attempt, max_retries, api_url, page, e,
            )
            if attempt < max_retries:
                time.sleep(1.5 * attempt)

    raise ScrapingError(f"Failed to fetch {api_url}?page={page}: {last_exc}")


# ---------------------------------------------------------------------------
# wikitext パーサ
# ---------------------------------------------------------------------------
# 見出し行: == .. == / === .. === / ==== .. ==== など（レベル1〜6）
_HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)

# テンプレート呼び出し {{...}}（入れ子1階層まで対応。ru.wiktionaryのテンプレートは
# 稀に {{выдел|...}} のような入れ子を含むため、非貪欲マッチで最短一致を狙う）
_TEMPLATE_RE = re.compile(r"\{\{([^{}]*(?:\{\{[^{}]*\}\}[^{}]*)*)\}\}")

# 内部リンク [[語|表示]] や [[語]]
_WIKILINK_RE = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]")

# '''強調''' や ''斜体''
_BOLD_ITALIC_RE = re.compile(r"'{2,5}")

# 定義行（# で始まる番号付きリスト = wiktionaryの語義番号）
_DEF_LINE_RE = re.compile(r"^#\s*(.+)$", re.MULTILINE)

# 箇条書き行（* または #）
_LIST_LINE_RE = re.compile(r"^[*#]\s*(.+)$", re.MULTILINE)


def _strip_wiki_markup(text: str) -> str:
    """wikitext断片から装飾記法を取り除き、プレーンテキストに近づける。"""
    # {{выдел|слово}} のような強調テンプレートは中身だけ残す
    text = re.sub(r"\{\{выдел\|([^}]*)\}\}", r"\1", text)
    # 未処理のテンプレートは丸ごと除去（{{семантика|...}} 等のメタ情報テンプレート）
    text = _TEMPLATE_RE.sub("", text)
    # 内部リンクは表示テキストのみ残す
    text = _WIKILINK_RE.sub(r"\1", text)
    # 太字・斜体マーカーを除去
    text = _BOLD_ITALIC_RE.sub("", text)
    # 参照タグなど残った不要マークアップを軽く除去
    text = re.sub(r"<ref[^>]*/?>.*?(</ref>)?", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def _split_sections(wikitext: str) -> list[dict]:
    """wikitextを見出し単位で分割する。
    戻り値: [{"level": int, "title": str, "body": str}, ...]
    先頭（最初の見出しより前）の部分は level=0, title="" として保持する。
    """
    matches = list(_HEADING_RE.finditer(wikitext))
    sections = []

    if not matches:
        return [{"level": 0, "title": "", "body": wikitext}]

    if matches[0].start() > 0:
        sections.append({"level": 0, "title": "", "body": wikitext[: matches[0].start()]})

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(wikitext)
        sections.append({"level": level, "title": title, "body": wikitext[body_start:body_end]})

    return sections


def _find_section_body(sections: list[dict], *title_candidates: str) -> Optional[str]:
    """見出しタイトルが候補のいずれかに一致（部分一致）するセクション本文を返す。
    最初に見つかったものを返す。"""
    for sec in sections:
        title_norm = sec["title"].strip()
        for cand in title_candidates:
            if cand.lower() in title_norm.lower():
                return sec["body"]
    return None


def _extract_definitions(body: str) -> list[str]:
    """「Значение」節本文から # で始まる語義定義を抽出し、装飾記法を除去して返す。"""
    results = []
    for m in _DEF_LINE_RE.finditer(body):
        line = m.group(1)
        cleaned = _strip_wiki_markup(line)
        if cleaned:
            results.append(cleaned)
    return results


def _extract_examples(body: str) -> list[str]:
    """{{пример|...}} テンプレートの第1引数（例文本文）を抽出する。
    {{пример|{{выдел|Слово}} от двери.}} のように内部に別テンプレートが
    入れ子になっているケースがあるため、波括弧の対応を数えて手動でスキャンする
    （単純な非貪欲正規表現では最初の内側の }} で止まってしまうため）。
    """
    results = []
    marker = "{{пример"
    pos = 0
    while True:
        start = body.find(marker, pos)
        if start == -1:
            break
        # マーカー直後から波括弧の深さを数えてテンプレート全体の終端を探す
        i = start + 2  # "{{" の直後
        depth = 1
        j = i
        while j < len(body) and depth > 0:
            if body.startswith("{{", j):
                depth += 1
                j += 2
            elif body.startswith("}}", j):
                depth -= 1
                j += 2
            else:
                j += 1
        inner = body[start + len(marker):j - 2]  # マーカー〜終端手前
        inner = inner[1:] if inner.startswith("|") else inner  # 先頭の "|" を除去
        # 名前付き引数（источник=... 等）は "|имя=" の形で始まるトップレベルの
        # 区切りとしてのみ切り離したいが、{{...}} 内部の | は無視する必要がある。
        # まずテンプレート/リンク記法をすべて展開・除去してから、素の "|" で分割する。
        cleaned_full = _strip_wiki_markup(inner)
        first_field = cleaned_full.split("|")[0]
        cleaned = first_field.strip()
        if cleaned:
            results.append(cleaned)
        pos = j
    return results


def _extract_list_items(body: str) -> list[str]:
    """* や # で始まる箇条書き（類語・コロケーション等）を抽出する。"""
    results = []
    for m in _LIST_LINE_RE.finditer(body):
        cleaned = _strip_wiki_markup(m.group(1))
        if cleaned:
            results.append(cleaned)
    return results


def _extract_pos_block(sections: list[dict]) -> list[str]:
    """「Морфологические и синтаксические свойства」節の本文（品詞テンプレート＋
    アクセント付き見出し行）から、テンプレートを除いたプレーンテキスト段落を返す。"""
    body = _find_section_body(
        sections,
        "Морфологические и синтаксические свойства",
        "Морфологические",
        "грамматические",
    )
    if not body:
        return []
    # 空行区切りで段落化し、テンプレート除去後に残る自然文のみ拾う
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    results = []
    for p in paragraphs:
        cleaned = _strip_wiki_markup(p)
        if cleaned:
            results.append(cleaned)
    return results


def _extract_accent_word(sections: list[dict]) -> list[str]:
    """品詞節の中の「'''слово́'''」のような太字強調（アクセント付き見出し語）を抽出する。"""
    body = _find_section_body(
        sections,
        "Морфологические и синтаксические свойства",
        "Морфологические",
    )
    if not body:
        return []
    results = []
    for m in re.finditer(r"'''([^']+)'''", body):
        text = m.group(1).strip()
        if text:
            results.append(text)
    return results


def _extract_etymology(sections: list[dict]) -> list[str]:
    body = _find_section_body(sections, "Этимология")
    if not body:
        return []
    cleaned = _strip_wiki_markup(body.strip())
    return [cleaned] if cleaned else []


def _extract_meaning_section(sections: list[dict]) -> Optional[str]:
    """「Значение」節（====レベル、多くは「Семантические свойства」の子節）の本文を返す。"""
    return _find_section_body(sections, "Значение")


# ---------------------------------------------------------------------------
# HTML方式の抽出（gufo.me / slovaronline / multitran / reverso 等）
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_ENTITY_MAP = {
    "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&#39;": "'", "&laquo;": "«", "&raquo;": "»",
    "&mdash;": "—", "&ndash;": "–", "&hellip;": "…",
}


def _strip_html_tags(fragment: str) -> str:
    """HTML断片からタグを除去し、主要なHTMLエンティティをデコードして整形する。"""
    text = _TAG_RE.sub(" ", fragment)
    for entity, repl in _ENTITY_MAP.items():
        text = text.replace(entity, repl)
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _extract_by_html_patterns(html: str, patterns: list[str]) -> list[str]:
    """patterns（正規表現文字列のリスト）を順に試し、最初にマッチが1件以上あった
    パターンの結果を採用する（サイトごとのDOM構造の揺れ・複数版フォールバック用）。
    各パターンは抽出したい範囲を1つの捕捉グループで囲んでおく想定。
    """
    for pattern in patterns:
        try:
            matches = re.findall(pattern, html, flags=re.IGNORECASE | re.DOTALL)
        except re.error as e:
            logger.warning("scraping: invalid html pattern=%s error=%s", pattern, e)
            continue
        if not matches:
            continue
        results = []
        for m in matches:
            raw = m if isinstance(m, str) else m[0]
            cleaned = _strip_html_tags(raw)
            if cleaned:
                results.append(cleaned)
        if results:
            return results
    return []


def apply_html_field_extractors(html: str, field_map: dict) -> dict:
    """field_map の値がパターンのリスト（例: config.json内で
    "field_map": {"translations": ["<pattern1>", "<pattern2>", ...]}）である場合に、
    HTMLへ順番にフォールバック適用して抽出する。

    値が単一文字列の場合は1要素のリストとして扱う。
    """
    extracted: dict = {}
    for field_name, patterns in field_map.items():
        pattern_list = patterns if isinstance(patterns, list) else [patterns]
        try:
            extracted[field_name] = _extract_by_html_patterns(html, pattern_list)
        except Exception as e:  # noqa: BLE001
            logger.warning("scraping: html extraction failed field=%s error=%s", field_name, e)
            extracted[field_name] = []
    return extracted


def apply_field_extractors(wikitext: str, field_map: dict) -> dict:
    """field_map（config.jsonのfield_map、キー=フィールド名、値=抽出種別）に従って
    wikitextから各フィールドを抽出する。

    field_map の値は以下のいずれかの識別子文字列:
      "meaning_definitions"  - Значение節の # 定義行
      "examples"              - {{пример|...}} の本文
      "pos_block"              - 品詞・文法情報節の平文
      "accent_word"            - 品詞節内の '''強調''' 見出し語
      "etymology"              - Этимология節の平文
      "synonyms_list"          - Синонимы節の箇条書き
      "collocation_list"       - Фразеологизмы/Синонимы節の箇条書き（用途に応じ節名を変更可）
    """
    sections = _split_sections(wikitext)
    meaning_body = _extract_meaning_section(sections) or ""

    extracted: dict = {}
    for field_name, extractor_id in field_map.items():
        try:
            if extractor_id == "meaning_definitions":
                extracted[field_name] = _extract_definitions(meaning_body)
            elif extractor_id == "examples":
                # 例文は Значение節配下だけでなくページ全体から拾う
                # （複数語義にまたがる {{пример}} を取りこぼさないため）
                extracted[field_name] = _extract_examples(wikitext)
            elif extractor_id == "pos_block":
                extracted[field_name] = _extract_pos_block(sections)
            elif extractor_id == "accent_word":
                extracted[field_name] = _extract_accent_word(sections)
            elif extractor_id == "etymology":
                extracted[field_name] = _extract_etymology(sections)
            elif extractor_id.startswith("list:"):
                # "list:Синонимы" のように section名を指定して箇条書きを拾う汎用指定子
                section_title = extractor_id.split(":", 1)[1]
                body = _find_section_body(sections, section_title) or ""
                extracted[field_name] = _extract_list_items(body)
            else:
                logger.warning("scraping: unknown extractor id=%s for field=%s", extractor_id, field_name)
                extracted[field_name] = []
        except Exception as e:  # noqa: BLE001
            logger.warning("scraping: extraction failed field=%s extractor=%s error=%s", field_name, extractor_id, e)
            extracted[field_name] = []

    return extracted


def scrape_source(word: str, source: dict, user_agent: str) -> dict:
    """1つのソース設定に基づいて1単語を取得・抽出する。
    source["source_type"] が "mediawiki_wikitext" ならwikitext方式、
    それ以外（未指定含む）ならHTML方式で処理する。

    戻り値: {"source_url": str, "status": "ok"|"not_found"|"error",
             "extracted": dict, "raw_html": str|None, "error": str|None}
    """
    source_type = source.get("source_type", "html")
    url = _build_url(source["url_template"], word)
    timeout = source.get("timeout_seconds", 15)
    max_retries = source.get("max_retries", 2)

    if source_type == "mediawiki_wikitext":
        api_url = _build_api_url(source["api_url_template"])
        try:
            raw_content = _fetch_wikitext(api_url, word, user_agent, timeout, max_retries)
        except ScrapingError as e:
            status = "not_found" if "404" in str(e) else "error"
            logger.info("scraping: word=%s source=%s status=%s (%s)", word, source["name"], status, e)
            return {"source_url": url, "status": status, "extracted": {}, "raw_html": None, "error": str(e)}

        try:
            extracted = apply_field_extractors(raw_content, source.get("field_map", {}))
        except Exception as e:  # noqa: BLE001
            logger.warning("scraping: wikitext parse error word=%s source=%s error=%s", word, source["name"], e)
            return {"source_url": url, "status": "error", "extracted": {}, "raw_html": raw_content, "error": str(e)}

    else:  # source_type == "html"（デフォルト）
        try:
            raw_content = _fetch_html_page(url, user_agent, timeout, max_retries)
        except ScrapingError as e:
            status = "not_found" if "404" in str(e) else "error"
            logger.info("scraping: word=%s source=%s status=%s (%s)", word, source["name"], status, e)
            return {"source_url": url, "status": status, "extracted": {}, "raw_html": None, "error": str(e)}

        try:
            extracted = apply_html_field_extractors(raw_content, source.get("field_map", {}))
        except Exception as e:  # noqa: BLE001
            logger.warning("scraping: html parse error word=%s source=%s error=%s", word, source["name"], e)
            return {"source_url": url, "status": "error", "extracted": {}, "raw_html": raw_content, "error": str(e)}

    return {
        "source_url": url,
        "status": "ok",
        "extracted": extracted,
        "raw_html": raw_content,
        "error": None,
    }


# ---------------------------------------------------------------------------
# キャッシュ付き取得（DB読み書き）
# ---------------------------------------------------------------------------
def _load_from_cache(db_path: str, word: str, source_url: str) -> Optional[dict]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT extracted, status FROM raw_data WHERE word = ? AND source_url = ?",
            (word, source_url),
        ).fetchone()
    if row is None:
        return None
    try:
        extracted = json.loads(row["extracted"])
    except (json.JSONDecodeError, TypeError):
        extracted = {}
    return {"status": row["status"], "extracted": extracted}


def _save_to_cache(db_path: str, word: str, result: dict, save_raw_html: bool = True) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO raw_data (word, source_url, raw_html, extracted, fetched_at, status)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(word, source_url) DO UPDATE SET
                raw_html = excluded.raw_html,
                extracted = excluded.extracted,
                fetched_at = excluded.fetched_at,
                status = excluded.status
            """,
            (
                word,
                result["source_url"],
                result["raw_html"] if save_raw_html else None,
                json.dumps(result["extracted"], ensure_ascii=False),
                now_iso(),
                result["status"],
            ),
        )
        conn.commit()


def scrape_word(word: str, config: dict) -> dict:
    """指定単語について config.scraping.sources に列挙された全ソースを処理する。
    各ソースはキャッシュがあれば再利用し、なければ取得してキャッシュに保存する。

    戻り値: {
        "word": word,
        "sources": {source_name: {"status": ..., "extracted": {...}}, ...}
    }
    """
    db_path = config["database"]["path"]
    user_agent = config["scraping"]["user_agent"]
    sources = [s for s in config["scraping"]["sources"] if s.get("enabled", True)]

    if not sources:
        raise ScrapingError("config.json の scraping.sources に有効なソースがありません。")

    results = {}
    for source in sources:
        url = _build_url(source["url_template"], word)
        cached = _load_from_cache(db_path, word, url)

        if cached is not None:
            logger.info("scraping: cache hit word=%s source=%s", word, source["name"])
            results[source["name"]] = cached
            continue

        result = scrape_source(word, source, user_agent)
        # 一時的なエラー（ネットワーク障害・レート制限等）はキャッシュしない。
        # "not_found"（404など、恒久的に情報が無いと判断できるもの）と
        # "ok" のみキャッシュし、"error" は次回実行時に再試行できるようにする。
        if result["status"] in ("ok", "not_found"):
            _save_to_cache(db_path, word, result)
        results[source["name"]] = {"status": result["status"], "extracted": result["extracted"]}

        delay = source.get("request_delay_seconds", 1.0)
        if delay:
            time.sleep(delay)

    return {"word": word, "sources": results}


import argparse
import json
import sys
# 必要な他のインポートはそのままにしてください
from common import ensure_db_initialized, load_config

def main():
    parser = argparse.ArgumentParser(description="単語リストをスクレイピングするツール")
    parser.add_argument("--startidx", type=int, default=1, help="開始行 (1始まり)")
    parser.add_argument("--endidx", type=int, default=100, help="終了行")
    parser.add_argument("--input", type=str, default="words.txt", help="入力ファイル名")
    
    args = parser.parse_args()

    # 設定読み込み・DB初期化
    cfg = load_config()
    ensure_db_initialized(cfg["database"]["path"])

    # ファイルの読み込みと範囲の抽出
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            # 1始まりのインデックスをリストのインデックスに合わせるため調整
            lines = [line.strip() for line in f.readlines()]
            target_words = lines[args.startidx - 1 : args.endidx]
    except FileNotFoundError:
        print(f"エラー: {args.input} が見つかりません。")
        sys.exit(1)

    # 処理実行
    results = []
    for word in target_words:
        if not word: continue
        print(f"Scraping: {word}")
        data = scrape_word(word, cfg)
        results.append(data)

    # 結果出力
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

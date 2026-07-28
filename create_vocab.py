"""
create_vocab.py
----------------
scraping.py が集めた生データ（raw_data）と summarize.py が構造化した
ロシア語データ（summaries）をもとに、無料枠のLLM API（Groq / Google AI Studio(Gemini) /
OpenRouter のいずれか、config.json の vocab_llm.provider で切替）を使って、
日本語学習者向けの単語帳データを生成し、CSVとDB（vocab_final テーブル）に保存する。

出力項目:
  ロシア語単語, アクセント, 品詞, 意味(日本語解説), 覚え方・コロケーション(日本語解説),
  例文(ロシア語), 例文訳(日本語), 類義語(ロシア語), 対義語(ロシア語),
  派生語・アスペクトペア(ロシア語)

使い方:
  python3 create_vocab.py --startidx 1 --endidx 10 --output vocab.csv
"""

import argparse
import csv
import hashlib
import json
import time

import requests

from common import ensure_db_initialized, get_connection, load_config, now_iso, setup_logger

logger = setup_logger("logs/errors.log")

ITEM_SEP = " / "
CSV_HEADER = [
    "Word", "Accent", "POS", "Gender", "Aspect", "PairedVerb",
    "Meaning_JA", "MemoryTip_JA",
    "Examples_RU", "Examples_JA",
    "Synonyms_RU", "Antonyms_RU", "RelatedWords_RU",
]

# ---------------------------------------------------------------------------
# LLM用プロンプト
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert Russian-language teacher creating a vocabulary card for a "
    "Japanese learner of Russian. You will be given a Russian dictionary entry: "
    "its natural, hand-picked meaning definitions, example sentences, and lists of "
    "synonyms/antonyms/related words scraped from Russian Wiktionary. These Russian "
    "materials are reliable and idiomatic. Your job is to write, in Japanese, a "
    "concise dictionary card based on them. Pay close attention to idioms, "
    "colloquial phrasing, and polysemous meanings in the Russian examples so your "
    "Japanese reflects the true intended meaning rather than a literal "
    "word-for-word translation. Do not invent facts not supported by the given "
    "Russian material. Output valid JSON only, with no markdown code fences."
)

USER_PROMPT_TEMPLATE = """\
# 単語
{word}（品詞: {pos}, 性: {gender}, 体: {aspect}, 完了体/不完了体ペア: {paired_verb}）

# ロシア語の意味定義（Wiktionaryより）
{meanings_ru}

# ロシア語の例文（Wiktionaryより、自然な文なので信頼できる）
{examples_ru}

# 既存のコロケーション（あれば）
{collocations_ru}

# 類義語（ロシア語）
{synonyms_ru}

# 対義語（ロシア語）
{antonyms_ru}

# 派生語・関連語（ロシア語）
{related_words_ru}

# タスク
上記のロシア語情報だけを根拠にして、日本語の単語帳カードを作成してください。
イディオムや多義語のニュアンスに注意し、直訳ではなく意図された意味を日本語で表現してください。

以下のJSON形式で出力してください（他のテキストは一切含めないこと）:
{{
  "Meaning_JA": "日本語での意味の解説。複数の意味がある場合は「 / 」区切りで並べる",
  "MemoryTip_JA": "語源・イメージ・語呂合わせなどの覚え方のコツと、代表的なコロケーション（よく使われる語の組み合わせ）を日本語で簡潔に説明する",
  "Examples_JA": "上のロシア語例文を1文ずつ日本語に訳したもの。ロシア語例文と同じ数、同じ順序で「 / 」区切りで並べる"
}}
"""


# ---------------------------------------------------------------------------
# DB読み込み: summaries + raw_data(ru_wiktionary の synonyms/antonyms/related_words)
# ---------------------------------------------------------------------------
def get_summary(db_path, word):
    """summaries テーブルから最新1件取得"""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM summaries WHERE word = ? ORDER BY created_at DESC, id DESC LIMIT 1",
            (word,),
        ).fetchone()
    if not row:
        return None
    return {
        k: (row[k] or "")
        for k in [
            "word", "pos", "gender", "aspect", "paired_verb",
            "meanings_ru", "collocations_ru", "examples_ru", "accent",
        ]
    }


def get_wiktionary_extras(db_path, word):
    """raw_data の ru_wiktionary エントリの extracted JSON から
    synonyms / antonyms / related_words を取得する。無ければ空リスト。"""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT extracted FROM raw_data "
            "WHERE word = ? AND source_url LIKE '%wiktionary%' AND status = 'ok' "
            "ORDER BY fetched_at DESC LIMIT 1",
            (word,),
        ).fetchone()
    result = {"synonyms": [], "antonyms": [], "related_words": []}
    if not row or not row["extracted"]:
        return result
    try:
        extracted = json.loads(row["extracted"])
    except (json.JSONDecodeError, TypeError):
        return result
    for key in result:
        values = extracted.get(key) or []
        # "#"やゴミ短片を除去し、実質的な語だけ残す
        cleaned = [v.strip() for v in values if v.strip() and v.strip() != "#"]
        result[key] = cleaned
    return result


# ---------------------------------------------------------------------------
# キャッシュ（vocab_final テーブル）
# ---------------------------------------------------------------------------
def ensure_vocab_final_table(db_path):
    """vocab_final テーブルが無ければ作成する（init_db.sqlにも定義済みだが、
    古いDBとの互換のためここでも保証する）。"""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vocab_final (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                word             TEXT NOT NULL,
                source_hash      TEXT NOT NULL,
                pos              TEXT,
                gender           TEXT,
                aspect           TEXT,
                paired_verb      TEXT,
                accent           TEXT,
                meaning_ja       TEXT,
                memory_tip_ja    TEXT,
                examples_ru      TEXT,
                examples_ja      TEXT,
                synonyms_ru      TEXT,
                antonyms_ru      TEXT,
                related_words_ru TEXT,
                provider         TEXT,
                model            TEXT,
                raw_llm_output   TEXT,
                created_at       TEXT NOT NULL,
                UNIQUE(word, source_hash)
            )
            """
        )
        conn.commit()


def compute_source_hash(ru, extras):
    payload = "\x1f".join([
        ru["meanings_ru"], ru["collocations_ru"], ru["examples_ru"],
        ITEM_SEP.join(extras["synonyms"]),
        ITEM_SEP.join(extras["antonyms"]),
        ITEM_SEP.join(extras["related_words"]),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_vocab_from_cache(db_path, word, source_hash):
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM vocab_final WHERE word = ? AND source_hash = ?",
            (word, source_hash),
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def save_vocab_to_cache(db_path, word, source_hash, ru, extras, generated, provider, model, raw_output):
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO vocab_final (
                word, source_hash, pos, gender, aspect, paired_verb, accent,
                meaning_ja, memory_tip_ja, examples_ru, examples_ja,
                synonyms_ru, antonyms_ru, related_words_ru,
                provider, model, raw_llm_output, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(word, source_hash) DO UPDATE SET
                pos = excluded.pos,
                gender = excluded.gender,
                aspect = excluded.aspect,
                paired_verb = excluded.paired_verb,
                accent = excluded.accent,
                meaning_ja = excluded.meaning_ja,
                memory_tip_ja = excluded.memory_tip_ja,
                examples_ru = excluded.examples_ru,
                examples_ja = excluded.examples_ja,
                synonyms_ru = excluded.synonyms_ru,
                antonyms_ru = excluded.antonyms_ru,
                related_words_ru = excluded.related_words_ru,
                provider = excluded.provider,
                model = excluded.model,
                raw_llm_output = excluded.raw_llm_output,
                created_at = excluded.created_at
            """,
            (
                word, source_hash, ru["pos"], ru["gender"], ru["aspect"], ru["paired_verb"], ru["accent"],
                generated["Meaning_JA"], generated["MemoryTip_JA"],
                ru["examples_ru"], generated["Examples_JA"],
                ITEM_SEP.join(extras["synonyms"]), ITEM_SEP.join(extras["antonyms"]),
                ITEM_SEP.join(extras["related_words"]),
                provider, model, raw_output, now_iso(),
            ),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# LLM API呼び出し（OpenAI互換 /chat/completions 形式。Groq / Gemini / OpenRouter 共通）
# ---------------------------------------------------------------------------
def call_llm_api(word, ru, provider_cfg):
    """OpenAI互換の /chat/completions エンドポイントを叩き、JSONをパースして返す。
    Groq・OpenRouterはネイティブ対応、GeminiもOpenAI互換パス(v1beta/openai)を使う。
    失敗時はリトライし、最終的に例外を送出する。"""
    prompt = USER_PROMPT_TEMPLATE.format(
        word=word,
        pos=ru["pos"] or "不明", gender=ru["gender"] or "-", aspect=ru["aspect"] or "-",
        paired_verb=ru["paired_verb"] or "-",
        meanings_ru=ru["meanings_ru"] or "-",
        examples_ru=ru["examples_ru"] or "-",
        collocations_ru=ru["collocations_ru"] or "-",
        synonyms_ru=ITEM_SEP.join(ru["_synonyms"]) or "-",
        antonyms_ru=ITEM_SEP.join(ru["_antonyms"]) or "-",
        related_words_ru=ITEM_SEP.join(ru["_related_words"]) or "-",
    )

    payload = {
        "model": provider_cfg["model"],
        "temperature": provider_cfg.get("temperature", 0.2),
        "max_tokens": provider_cfg.get("max_tokens", 2048),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {provider_cfg['api_key']}",
        "Content-Type": "application/json",
    }
    url = f"{provider_cfg['base_url'].rstrip('/')}/chat/completions"
    max_retries = max(1, provider_cfg.get("max_retries", 3))
    timeout = provider_cfg.get("timeout_seconds", 60)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=timeout)
            res.raise_for_status()
            content = res.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            for key in ("Meaning_JA", "MemoryTip_JA", "Examples_JA"):
                if key not in parsed:
                    raise ValueError(f"LLM応答に必須キー '{key}' が含まれていません")
            return parsed, content
        except Exception as e:  # noqa: BLE001
            last_error = e
            logger.warning(
                "create_vocab: LLM API呼び出しに失敗 (attempt %d/%d) word=%s error=%s",
                attempt, max_retries, word, e,
            )
            if attempt < max_retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"LLM API呼び出しに失敗しました: word={word}") from last_error


# ---------------------------------------------------------------------------
# オーケストレーション
# ---------------------------------------------------------------------------
def get_provider_config(cfg):
    vocab_llm_cfg = cfg["vocab_llm"]
    provider_name = vocab_llm_cfg["provider"]
    providers = vocab_llm_cfg.get("providers", {})
    if provider_name not in providers:
        raise KeyError(
            f"vocab_llm.provider='{provider_name}' が vocab_llm.providers に定義されていません。"
            f" 定義済み: {list(providers.keys())}"
        )
    provider_cfg = providers[provider_name]
    if not provider_cfg.get("api_key") or provider_cfg["api_key"].startswith("YOUR_"):
        raise ValueError(
            f"vocab_llm.providers.{provider_name}.api_key が未設定です。config.json に実際のAPIキーを設定してください。"
        )
    return provider_name, provider_cfg


def build_vocab_entry(word, db_path, provider_name, provider_cfg, delay=0.5):
    ru = get_summary(db_path, word)
    if not ru:
        return None, "summariesにデータが無いためスキップ"

    extras = get_wiktionary_extras(db_path, word)
    source_hash = compute_source_hash(ru, extras)

    cached = load_vocab_from_cache(db_path, word, source_hash)
    if cached is not None:
        return cached, None

    ru["_synonyms"] = extras["synonyms"]
    ru["_antonyms"] = extras["antonyms"]
    ru["_related_words"] = extras["related_words"]

    try:
        generated, raw_output = call_llm_api(word, ru, provider_cfg)
    except Exception as e:  # noqa: BLE001
        return None, str(e)

    save_vocab_to_cache(
        db_path, word, source_hash, ru, extras, generated,
        provider=provider_name, model=provider_cfg["model"], raw_output=raw_output,
    )
    time.sleep(delay)

    row = load_vocab_from_cache(db_path, word, source_hash)
    return row, None


def row_to_csv_dict(row):
    return {
        "Word": row["word"],
        "Accent": row["accent"] or "",
        "POS": row["pos"] or "",
        "Gender": row["gender"] or "",
        "Aspect": row["aspect"] or "",
        "PairedVerb": row["paired_verb"] or "",
        "Meaning_JA": row["meaning_ja"] or "",
        "MemoryTip_JA": row["memory_tip_ja"] or "",
        "Examples_RU": row["examples_ru"] or "",
        "Examples_JA": row["examples_ja"] or "",
        "Synonyms_RU": row["synonyms_ru"] or "",
        "Antonyms_RU": row["antonyms_ru"] or "",
        "RelatedWords_RU": row["related_words_ru"] or "",
    }


def write_csv(rows, output_file, use_bom=True):
    encoding = "utf-8-sig" if use_bom else "utf-8"
    with open(output_file, "w", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="summaries/raw_data から、外部LLM API(Groq/Gemini/OpenRouter)を使って"
                    "日本語の単語帳データを生成し、CSVとDB(vocab_final)に保存するツール"
    )
    parser.add_argument("--startidx", type=int, default=1, help="開始行 (1始まり)")
    parser.add_argument("--endidx", type=int, default=100, help="終了行")
    parser.add_argument("--input", type=str, default="words.txt", help="入力ファイル名")
    parser.add_argument("--output", type=str, default=None, help="出力CSVファイル名（省略時は config.json の設定値）")
    parser.add_argument(
        "--provider", type=str, default=None,
        help="使用するAPIプロバイダ名（groq/gemini/openrouter等）。省略時は config.json の vocab_llm.provider",
    )
    args = parser.parse_args()

    cfg = load_config()
    db_path = cfg["database"]["path"]
    ensure_db_initialized(db_path)
    ensure_vocab_final_table(db_path)

    if args.provider:
        cfg["vocab_llm"]["provider"] = args.provider
    provider_name, provider_cfg = get_provider_config(cfg)
    delay = provider_cfg.get("request_delay_seconds", 0.5)

    output_file = args.output or cfg["pipeline"]["output_file"]
    use_bom = cfg["pipeline"]["csv_bom"]

    print(f"単語帳生成モード: provider={provider_name}, model={provider_cfg['model']}")

    with open(args.input, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    start = max(0, args.startidx - 1)
    end = min(len(words), args.endidx)
    target_words = words[start:end]

    rows = []
    for i, word in enumerate(target_words, start=start + 1):
        print(f"[{i}/{len(words)}] 処理中: {word}")
        row, error = build_vocab_entry(word, db_path, provider_name, provider_cfg, delay=delay)
        if error:
            print(f"  -> スキップ: {error}")
            logger.warning("create_vocab: word=%s error=%s", word, error)
            continue
        rows.append(row_to_csv_dict(row))

    write_csv(rows, output_file, use_bom=use_bom)
    print(f"完了: {len(rows)}件を '{output_file}' に書き出しました。")


if __name__ == "__main__":
    main()

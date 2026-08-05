"""
create_vocab.py
----------------
scraping.py が raw_data テーブルに保存した生データ（主に ru.wiktionary.org から
抽出したロシア語の意味・例文・品詞情報・類義語/対義語/派生語など）を直接読み込み、
LLM API（Ollama等のOpenAI互換エンドポイント。config.json の vocab_llm.provider で切替）を
1回呼び出すだけで、日本語学習者向けの単語帳データを生成し、CSVとDB（vocab_final テーブル）に保存する。

【ハルシネーション対策】
- 例文原文・類義語・対義語・派生語・アクセントは raw_data からそのまま決定的に転記し、
  LLMには生成させない。
- LLMに生成させるのは「日本語での説明・翻訳」という言語表現部分のみに限定する。
- Meaning_JAは単語帳の選択肢として使うため、最大3項目まで決定的に切り詰める
  （各項目の文字数はLLMへの指示のみに委ね、コード側での文字カットはしない）。
- Examples_JAはExamples_RUと同じ数・同じ順序の" / "区切り項目になるようプロンプトで
  指示し、項目数が一致しない場合はその単語の生成全体を失敗として扱う。

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
    "Examples_RU", "Examples_JA", "Examples_RU_Source",
    "Synonyms_RU", "Antonyms_RU", "RelatedWords_RU",
]

MAX_MEANING_ITEMS = 3
MAX_EXAMPLES = 2


class VocabGenerationError(Exception):
    pass


# ---------------------------------------------------------------------------
# raw_data からのロシア語データ読み込み
# ---------------------------------------------------------------------------
def _get_raw_extracted(db_path: str, word: str, source_url_substr: str) -> dict | None:
    """raw_data から、指定した単語・ソース（URLの部分一致）の最新かつ status='ok' な
    extracted(JSON)を取得する。見つからなければ None。"""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT extracted FROM raw_data "
            "WHERE word = ? AND source_url LIKE ? AND status = 'ok' "
            "ORDER BY fetched_at DESC LIMIT 1",
            (word, f"%{source_url_substr}%"),
        ).fetchone()
    if not row or not row["extracted"]:
        return None
    try:
        return json.loads(row["extracted"])
    except (json.JSONDecodeError, TypeError):
        return None


def _normalize_joined(s: str) -> str:
    """" / "区切り文字列から、空項目・前後の余分な空白を除去して再結合する。"""
    if not isinstance(s, str):
        return s
    return ITEM_SEP.join(t.strip() for t in s.split(ITEM_SEP) if t.strip())


# Wiktionaryが「該当データなし/不明」を表すために使うプレースホルダー記号。
_PLACEHOLDER_TOKENS = {"—", "–", "-", "?", "#", "??", "###"}


def _join_items(items) -> str:
    """extracted の値（文字列のリスト）を、空要素・プレースホルダーのみの要素を除いて
    " / " で結合する。"""
    if not items:
        return ""
    return ITEM_SEP.join(
        t.strip() for t in items
        if isinstance(t, str) and t.strip() and t.strip() not in _PLACEHOLDER_TOKENS
    )


def get_ru_source_data(db_path: str, word: str) -> dict | None:
    """raw_data から、ロシア語の一次情報（ru_wiktionary）と、補助的な参考情報
    （multitran / reverso_context の英訳、あれば）をまとめて取得する。"""
    wiktionary = _get_raw_extracted(db_path, word, "wiktionary")
    if wiktionary is None:
        return None

    multitran = _get_raw_extracted(db_path, word, "multitran") or {}
    reverso = _get_raw_extracted(db_path, word, "reverso") or {}

    return {
        "meanings_ru": _join_items(wiktionary.get("meaning_list")),
        "examples_ru": _join_items(wiktionary.get("example_list")),
        "pos_block_ru": _join_items(wiktionary.get("pos_block")),
        "accent": _join_items(wiktionary.get("accent_word")),
        "collocations_ru": _join_items(wiktionary.get("collocation_list")),
        "synonyms_ru": _join_items(wiktionary.get("synonyms")),
        "antonyms_ru": _join_items(wiktionary.get("antonyms")),
        "related_words_ru": _join_items(wiktionary.get("related_words")),
        "etymology_ru": _join_items(wiktionary.get("etymology")),
        "multitran_translations_en": _join_items(multitran.get("translations")),
        "reverso_translations_en": _join_items(reverso.get("translations")),
    }


# ---------------------------------------------------------------------------
# LLM用プロンプト（短縮版）
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Russian teacher making Japanese vocab flashcards from Wiktionary data. "
    "Input: grammar text, RU meaning defs, existing RU examples (reference only, "
    "don't translate verbatim), collocations, synonyms/antonyms/related words, "
    "optional noisy EN glosses. "
    "Tasks: "
    "(1) POS, gender (nouns), aspect+paired verb (verbs); empty if N/A. "
    "(2) Meaning_JA: up to 3 short JA gloss words (~10 chars each, dictionary-"
    "headword style, NOT sentences), most common senses first. "
    "MemoryTip_JA: JA mnemonic + common collocations. "
    "(3) Write 2 NEW natural modern RU example sentences (main senses only, not "
    "verbatim from input) + JA translations. "
    "No invented meanings/grammar beyond given material. Output raw JSON only, "
    "no fences/commentary. In ' / '-joined fields: no empty items, no trailing ' / '."
)

_JSON_INSTRUCTION = """\
JSON形式のみで出力（他のテキスト禁止）:
{
  "POS": "品詞", "Gender": "性(名詞のみ)", "Aspect": "体(動詞のみ)",
  "PairedVerb": "ペア動詞(動詞のみ)",
  "Meaning_JA": "短い訳語、最大3項目、各10字程度、「 / 」区切り",
  "MemoryTip_JA": "覚え方+コロケーション(日本語)",
  "Examples_RU": "新規ロシア語例文(最大2文)、「 / 」区切り",
  "Examples_JA": "Examples_RUと同数同順の日本語訳、「 / 」区切り"
}"""

# フィールド名 -> raw_data辞書のキー・プロンプト内ラベルの対応。
# 値が空のフィールドはプロンプトに含めないことで入力トークンを削減する。
_FIELD_LABELS = [
    ("pos_block_ru", "文法情報"),
    ("meanings_ru", "意味定義"),
    ("examples_ru", "既存例文(参考のみ,直訳しない)"),
    ("collocations_ru", "コロケーション"),
    ("etymology_ru", "語源"),
    ("synonyms_ru", "類義語"),
    ("antonyms_ru", "対義語"),
    ("related_words_ru", "関連語"),
]


def build_user_prompt(word: str, ru: dict) -> str:
    """raw_dataの中身に応じて、値が入っているフィールドだけを組み立てて
    ユーザープロンプトを生成する（空フィールドは省略してトークンを節約）。"""
    lines = [f"単語: {word}"]
    for key, label in _FIELD_LABELS:
        val = ru.get(key)
        if val:
            lines.append(f"{label}: {val}")

    en_parts = []
    if ru.get("multitran_translations_en"):
        en_parts.append(f"Multitran: {ru['multitran_translations_en']}")
    if ru.get("reverso_translations_en"):
        en_parts.append(f"Reverso: {ru['reverso_translations_en']}")
    if en_parts:
        lines.append("英訳(参考程度、ノイズ含む可能性あり): " + " / ".join(en_parts))

    lines.append("")
    lines.append(_JSON_INSTRUCTION)
    return "\n".join(lines)


_REQUIRED_KEYS = ("POS", "Gender", "Aspect", "PairedVerb", "Meaning_JA", "MemoryTip_JA", "Examples_RU", "Examples_JA")


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
                examples_ru_source TEXT,
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
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(vocab_final)")}
        if "examples_ru_source" not in existing_cols:
            conn.execute("ALTER TABLE vocab_final ADD COLUMN examples_ru_source TEXT")

        # 複数プロセス並列実行時、同じ単語をLLMに二重処理させないための予約テーブル。
        # INSERT OR IGNORE がSQLiteのUNIQUE制約でアトミックに競合を防ぐ。
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vocab_in_progress (
                word         TEXT NOT NULL,
                source_hash  TEXT NOT NULL,
                started_at   TEXT NOT NULL,
                PRIMARY KEY (word, source_hash)
            )
            """
        )
        conn.commit()


def try_acquire_lock(db_path, word, source_hash) -> bool:
    """他プロセスがまだ処理していなければ予約行を挿入してTrueを返す。
    既に他プロセスが処理中（行が既に存在）ならFalseを返す。"""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO vocab_in_progress (word, source_hash, started_at) VALUES (?, ?, ?)",
            (word, source_hash, now_iso()),
        )
        conn.commit()
        return cur.rowcount > 0


def release_lock(db_path, word, source_hash):
    with get_connection(db_path) as conn:
        conn.execute(
            "DELETE FROM vocab_in_progress WHERE word = ? AND source_hash = ?",
            (word, source_hash),
        )
        conn.commit()


def compute_source_hash(ru: dict) -> str:
    payload = "\x1f".join([
        ru["meanings_ru"], ru["examples_ru"], ru["pos_block_ru"], ru["accent"],
        ru["collocations_ru"], ru["etymology_ru"],
        ru["synonyms_ru"], ru["antonyms_ru"], ru["related_words_ru"],
        ru["multitran_translations_en"], ru["reverso_translations_en"],
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


def save_vocab_to_cache(db_path, word, source_hash, ru, generated, provider, model, raw_output):
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO vocab_final (
                word, source_hash, pos, gender, aspect, paired_verb, accent,
                meaning_ja, memory_tip_ja, examples_ru, examples_ja, examples_ru_source,
                synonyms_ru, antonyms_ru, related_words_ru,
                provider, model, raw_llm_output, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                examples_ru_source = excluded.examples_ru_source,
                synonyms_ru = excluded.synonyms_ru,
                antonyms_ru = excluded.antonyms_ru,
                related_words_ru = excluded.related_words_ru,
                provider = excluded.provider,
                model = excluded.model,
                raw_llm_output = excluded.raw_llm_output,
                created_at = excluded.created_at
            """,
            (
                word, source_hash, generated["POS"], generated["Gender"],
                generated["Aspect"], generated["PairedVerb"], ru["accent"],
                generated["Meaning_JA"], generated["MemoryTip_JA"],
                generated["Examples_RU"], generated["Examples_JA"], ru["examples_ru"],
                ru["synonyms_ru"], ru["antonyms_ru"], ru["related_words_ru"],
                provider, model, raw_output, now_iso(),
            ),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# LLM API呼び出し（OpenAI互換 /chat/completions 形式。Ollama等）
# ---------------------------------------------------------------------------
def call_llm_api(word: str, ru: dict, provider_cfg: dict) -> tuple[dict, str]:
    """OpenAI互換の /chat/completions エンドポイントを叩き、JSONをパースして返す。
    失敗時（HTTPエラー・JSONパース失敗・必須キー欠落・例文の項目数不一致）はリトライし、
    最終的に VocabGenerationError を送出する。"""
    prompt = build_user_prompt(word, ru)

    payload = {
        "model": provider_cfg["model"],
        "temperature": provider_cfg.get("temperature", 0.2),
        "max_tokens": provider_cfg.get("max_tokens", 1024),
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

            for key in _REQUIRED_KEYS:
                if key not in parsed or not isinstance(parsed[key], str):
                    raise ValueError(f"LLM応答に必須キー '{key}'（文字列）が含まれていません")

            # プロンプトの指示（空項目・末尾区切り禁止）だけに頼らず、決定的に正規化する
            for key in ("Meaning_JA", "Examples_RU", "Examples_JA"):
                parsed[key] = _normalize_joined(parsed[key])

            # Meaning_JAは単語帳の選択肢として使うため項目数だけ決定的に制限する
            # （字数はプロンプトの指示("~10字程度")のみに委ね、文字列カットはしない）
            meaning_items = [t for t in parsed["Meaning_JA"].split(ITEM_SEP) if t]
            parsed["Meaning_JA"] = ITEM_SEP.join(meaning_items[:MAX_MEANING_ITEMS])

            # 例文も最大2件までという指示を、プロンプト遵守だけに頼らず決定的に切り詰める
            for key in ("Examples_RU", "Examples_JA"):
                items = [t for t in parsed[key].split(ITEM_SEP) if t]
                parsed[key] = ITEM_SEP.join(items[:MAX_EXAMPLES])

            ru_count = len([t for t in parsed["Examples_RU"].split(ITEM_SEP) if t])
            ja_count = len([t for t in parsed["Examples_JA"].split(ITEM_SEP) if t])
            if ru_count == 0:
                raise ValueError("Examples_RUが生成されていません")
            if ru_count != ja_count:
                raise ValueError(f"例文の項目数が不一致（RU={ru_count}, JA={ja_count}）")

            return parsed, content
        except Exception as e:  # noqa: BLE001
            last_error = e
            logger.warning(
                "create_vocab: LLM API呼び出しに失敗 (attempt %d/%d) word=%s error=%s",
                attempt, max_retries, word, e,
            )
            if attempt < max_retries:
                time.sleep(2 * attempt)

    raise VocabGenerationError(f"LLM API呼び出しに失敗しました: word={word}") from last_error


# ---------------------------------------------------------------------------
# オーケストレーション
# ---------------------------------------------------------------------------
def get_provider_config(cfg, override_api_key=None):
    vocab_llm_cfg = cfg["vocab_llm"]
    provider_name = vocab_llm_cfg["provider"]
    providers = vocab_llm_cfg.get("providers", {})
    if provider_name not in providers:
        raise KeyError(f"config.jsonにプロバイダ '{provider_name}' の設定がありません")
    provider_cfg = providers[provider_name]

    if override_api_key:
        provider_cfg["api_key"] = override_api_key

    if not provider_cfg.get("api_key"):
        raise ValueError(f"プロバイダ '{provider_name}' の api_key が設定されていません")
    return provider_name, provider_cfg


def build_vocab_entry(word, db_path, provider_name, provider_cfg, delay=0.5):
    ru = get_ru_source_data(db_path, word)
    if ru is None:
        return None, "raw_data(ru_wiktionary)にデータが無いためスキップ（先にscraping.pyを実行してください）"

    source_hash = compute_source_hash(ru)

    cached = load_vocab_from_cache(db_path, word, source_hash)
    if cached is not None:
        return cached, None

    # 他プロセスが同じ単語を処理中なら、LLM呼び出しをせずスキップする
    if not try_acquire_lock(db_path, word, source_hash):
        return None, "他プロセスが処理中のためスキップ"

    try:
        try:
            generated, raw_output = call_llm_api(word, ru, provider_cfg)
        except VocabGenerationError as e:
            return None, str(e)

        save_vocab_to_cache(
            db_path, word, source_hash, ru, generated,
            provider=provider_name, model=provider_cfg["model"], raw_output=raw_output,
        )
    finally:
        # ロックはLLM呼び出し+保存が終わり次第すぐ解放する。
        # delay(APIレート制限用の待機)はロック保持と無関係なので、解放後に行う
        # ＝他プロセスは自分の待機を待たされず、次の単語の処理に進める。
        release_lock(db_path, word, source_hash)

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
        "Examples_RU_Source": row["examples_ru_source"] or "",
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
        description="raw_data(scraping.pyの取得結果)から、LLM API(Ollama等)を使って"
                    "日本語の単語帳データを直接生成し、CSVとDB(vocab_final)に保存するツール"
    )
    parser.add_argument("--startidx", type=int, default=1, help="開始行 (1始まり)")
    parser.add_argument("--endidx", type=int, default=100, help="終了行")
    parser.add_argument("--input", type=str, default="words.txt", help="入力ファイル名")
    parser.add_argument("--output", type=str, default=None, help="出力CSVファイル名（省略時は config.json の設定値）")
    parser.add_argument(
        "--provider", type=str, default=None,
        help="使用するAPIプロバイダ名（ollama等）。省略時は config.json の vocab_llm.provider",
    )
    parser.add_argument("--api-key", type=str, default=None, help="LLM APIキー（config.jsonより優先）")

    args = parser.parse_args()

    cfg = load_config()
    db_path = cfg["database"]["path"]
    ensure_db_initialized(db_path)
    ensure_vocab_final_table(db_path)

    if args.provider:
        cfg["vocab_llm"]["provider"] = args.provider
    provider_name, provider_cfg = get_provider_config(cfg, override_api_key=args.api_key)
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

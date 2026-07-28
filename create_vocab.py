"""
create_vocab.py
----------------
scraping.py が raw_data テーブルに保存した生データ（主に ru.wiktionary.org から
抽出したロシア語の意味・例文・品詞情報・類義語/対義語/派生語など）を直接読み込み、
無料枠のLLM API（Groq / Google AI Studio(Gemini) / OpenRouter のいずれか、
config.json の vocab_llm.provider で切替）を1回呼び出すだけで、
日本語学習者向けの単語帳データを生成し、CSVとDB（vocab_final テーブル）に保存する。

【中間のLLM構造化ステージ（旧 summarize.py）は存在しない】
以前のバージョンでは、ロシア語の生データをローカルLLM(Ollama)でいったん
品詞/性/体/意味/例文などに構造化する summarize.py を挟んでいたが、
本バージョンでは廃止した。Groq/Gemini等の外部API側のモデルは十分に高性能なため、
raw_data の生テキスト（wikitextから軽くクリーニングしただけの品詞節・語義節・
例文など）をそのままプロンプトに渡し、1回のAPI呼び出しで
  (a) 品詞・性・体・ペア動詞などの文法情報の抽出
  (b) 日本語での意味・覚え方の解説の生成
  (c) 例文の日本語訳
をまとめて行わせる。

【ハルシネーション対策】
- 意味・品詞情報の"事実"に関わる部分（例文原文・類義語・対義語・派生語・アクセント）は
  raw_data からそのまま決定的に転記し、LLMには生成させない。
- LLMに生成させるのは「日本語での説明・翻訳」という言語表現部分のみに限定する。
- 例文の日本語訳(Examples_JA)は、原文(Examples_RU)と同じ数・同じ順序の
  " / "区切り項目になるようプロンプトで指示し、項目数が一致しない場合は
  その単語の生成全体を失敗として扱う（中途半端な対応関係のデータをDB/CSVに
  混入させないためのフェイルセーフ）。

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
    "Examples_RU", "Examples_JA", "Examples_RU_Source",
    "Synonyms_RU", "Antonyms_RU", "RelatedWords_RU",
]


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
    """" / "区切り文字列から、空項目・前後の余分な空白を除去して再結合する。
    プロンプト遵守（空項目/末尾区切り禁止の指示）だけに頼らず、決定的に保証するためのフェイルセーフ。"""
    if not isinstance(s, str):
        return s
    return ITEM_SEP.join(t.strip() for t in s.split(ITEM_SEP) if t.strip())


# Wiktionaryが「該当データなし/不明」を表すために使うプレースホルダー記号。
# これらのみで構成される項目は情報を持たないため、_join_items で除去する。
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
    （multitran / reverso_context の英訳、あれば）をまとめて取得する。
    ru_wiktionary のデータが無ければ単語帳カードを作れないため None を返す。"""
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
        # 補助的な参考情報（英訳）。無ければ空文字のまま、プロンプトには「参考程度」として渡す。
        "multitran_translations_en": _join_items(multitran.get("translations")),
        "reverso_translations_en": _join_items(reverso.get("translations")),
    }


# ---------------------------------------------------------------------------
# LLM用プロンプト
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert Russian-language teacher creating a vocabulary flashcard for a "
    "Japanese learner of Russian, to be studied in a spaced-repetition app. You will be "
    "given raw data scraped from Russian Wiktionary for a single word: a paragraph "
    "describing its grammatical properties (part of speech, gender, aspect, aspectual "
    "pair, etc.), its natural, hand-picked meaning definitions, some existing example "
    "sentences (these are real citations from a Russian corpus — reliable but often in "
    "a literary/formal register), and lists of synonyms/antonyms/related words. This "
    "Russian material is reliable and idiomatic, though the grammar paragraph is "
    "unstructured free text that you must parse yourself, and the existing example "
    "sentences may be sparse, missing, or stylistically dated for a learner. Optionally "
    "you may also receive short English translations from other dictionaries as extra "
    "reference (these can be noisy or incomplete; trust the Russian material first). "
    "Your job is three-fold: "
    "(1) parse the grammar paragraph to identify part of speech, gender (nouns only), "
    "aspect (verbs only), and the aspectual pair verb (verbs only) — leave a field "
    "empty if it does not apply or is not stated; "
    "(2) write, in Japanese, a concise dictionary card based on the Russian material: "
    "the meaning and a memory tip / mnemonic / common collocations; "
    "(3) instead of translating the existing example sentences, WRITE YOUR OWN new, "
    "natural, contemporary Russian example sentences — at most two, covering the most "
    "representative meaning sense(s) from Meaning_JA (if the word is polysemous, pick "
    "the one or two most important/common senses rather than covering every sense) — "
    "plus their Japanese translations. Each generated example "
    "must clearly illustrate one of the meaning senses you identified; do not invent a "
    "meaning or usage that is not supported by the given Russian material (definitions, "
    "existing examples, or collocations). Prefer everyday, natural phrasing a modern "
    "native speaker would actually say over rare vocabulary or ornate literary "
    "constructions, while staying grammatically correct and idiomatic. You may use the "
    "existing example sentences as inspiration/reference but should not reuse them "
    "verbatim. "
    "Pay close attention to idioms, colloquial phrasing, and polysemous meanings so "
    "your Japanese reflects the true intended meaning rather than a literal "
    "word-for-word translation. Do not invent grammatical facts or meanings not "
    "supported by the given Russian material. Output valid JSON only, with no "
    "markdown code fences. Never include empty items or trailing ' / ' separators in "
    "any of the ' / '-joined string fields."
)

USER_PROMPT_TEMPLATE = """\
# 単語
{word}

# 文法情報（ru.wiktionary.orgの品詞節、未整形の自然文。ここから品詞/性/体/ペア動詞を判定する）
{pos_block_ru}

# ロシア語の意味定義（Wiktionaryより）
{meanings_ru}

# 既存の例文（Wiktionaryより、参考情報。実在するコーパスからの引用で内容は信頼できるが、
# やや硬い/古い文体のことがある。そのまま訳すのではなく、意味の把握と新しい例文作成の参考にすること）
{examples_ru}

# 既存のコロケーション（あれば）
{collocations_ru}

# 語源（あれば、覚え方の参考に）
{etymology_ru}

# 類義語（ロシア語、参考情報）
{synonyms_ru}

# 対義語（ロシア語、参考情報）
{antonyms_ru}

# 派生語・関連語（ロシア語、参考情報）
{related_words_ru}

# 他辞書からの英訳（参考程度、ノイズを含む可能性あり）
Multitran: {multitran_translations_en}
Reverso Context: {reverso_translations_en}

# タスク
上記のロシア語情報を根拠にして、日本語の単語帳カードを作成してください。
イディオムや多義語のニュアンスに注意し、直訳ではなく意図された意味を日本語で表現してください。
文法情報は「文法情報」欄の自然文から読み取り、記載が無い項目は空文字にしてください。
例文は「既存の例文」をそのまま訳すのではなく、Meaning_JAで挙げた意味の中から代表的な
意味を最大2つ選び、それぞれに対応する自然で現代的な新しいロシア語例文（合計最大2文）を
あなた自身が作成し、その日本語訳も付けてください（多義語のすべての意味に例文をつける
必要はありません。意味を勝手に増やして例文を作らないこと。既存の例文にある用法・
コロケーションの範囲内で自然な文を作ること）。

以下のJSON形式で出力してください（他のテキストは一切含めないこと）:
{{
  "POS": "品詞（例: 名詞, 動詞, 形容詞 など。文法情報欄から判断）",
  "Gender": "性（名詞の場合のみ。例: 男性, 女性, 中性。該当しなければ空文字）",
  "Aspect": "体（動詞の場合のみ。例: 完了体, 不完了体。該当しなければ空文字）",
  "PairedVerb": "完了体/不完了体のペア動詞（動詞の場合のみ、ロシア語表記。無ければ空文字）",
  "Meaning_JA": "日本語での意味の解説。複数の意味がある場合は「 / 」区切りで並べる。空項目・末尾の区切りは禁止",
  "MemoryTip_JA": "語源・イメージ・語呂合わせなどの覚え方のコツと、代表的なコロケーション（よく使われる語の組み合わせ）を日本語で簡潔に説明する",
  "Examples_RU": "代表的な意味（最大2つ）についてあなたが新規作成した自然なロシア語例文（最大2文）。「 / 」区切りで並べる。空項目・末尾の区切りは禁止",
  "Examples_JA": "Examples_RUと同じ数・同じ順序の日本語訳を「 / 」区切りで並べる。空項目・末尾の区切りは禁止"
}}
"""

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
        # 既存DB（CREATE TABLE IF NOT EXISTSでは列追加されない）向けのマイグレーション
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(vocab_final)")}
        if "examples_ru_source" not in existing_cols:
            conn.execute("ALTER TABLE vocab_final ADD COLUMN examples_ru_source TEXT")
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
# LLM API呼び出し（OpenAI互換 /chat/completions 形式。Groq / Gemini / OpenRouter 共通）
# ---------------------------------------------------------------------------
def call_llm_api(word: str, ru: dict, provider_cfg: dict) -> tuple[dict, str]:
    """OpenAI互換の /chat/completions エンドポイントを叩き、JSONをパースして返す。
    Groq・OpenRouterはネイティブ対応、GeminiもOpenAI互換パス(v1beta/openai)を使う。
    失敗時（HTTPエラー・JSONパース失敗・必須キー欠落・例文の項目数不一致）はリトライし、
    最終的に VocabGenerationError を送出する。"""
    prompt = USER_PROMPT_TEMPLATE.format(
        word=word,
        pos_block_ru=ru["pos_block_ru"] or "-",
        meanings_ru=ru["meanings_ru"] or "-",
        examples_ru=ru["examples_ru"] or "-",
        collocations_ru=ru["collocations_ru"] or "-",
        etymology_ru=ru["etymology_ru"] or "-",
        synonyms_ru=ru["synonyms_ru"] or "-",
        antonyms_ru=ru["antonyms_ru"] or "-",
        related_words_ru=ru["related_words_ru"] or "-",
        multitran_translations_en=ru["multitran_translations_en"] or "-",
        reverso_translations_en=ru["reverso_translations_en"] or "-",
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

            for key in _REQUIRED_KEYS:
                if key not in parsed or not isinstance(parsed[key], str):
                    raise ValueError(f"LLM応答に必須キー '{key}'（文字列）が含まれていません")

            # プロンプトの指示（空項目・末尾区切り禁止）だけに頼らず、決定的に正規化する
            for key in ("Meaning_JA", "Examples_RU", "Examples_JA"):
                parsed[key] = _normalize_joined(parsed[key])

            # 例文は最大2件までという指示も、プロンプト遵守だけに頼らず決定的に切り詰める
            MAX_EXAMPLES = 2
            for key in ("Examples_RU", "Examples_JA"):
                items = [t for t in parsed[key].split(ITEM_SEP) if t.strip()]
                parsed[key] = ITEM_SEP.join(items[:MAX_EXAMPLES])

            # 例文は今回LLMが新規生成するため、比較対象はスクレイピング原文ではなく
            # 生成したExamples_RUとExamples_JA同士。対応関係が崩れている（訳の欠落・混入）
            # 疑いがある場合は生成失敗として扱う
            generated_ru_count = len([t for t in parsed["Examples_RU"].split(ITEM_SEP) if t.strip()])
            generated_ja_count = len([t for t in parsed["Examples_JA"].split(ITEM_SEP) if t.strip()])
            if generated_ru_count == 0:
                raise ValueError("Examples_RUが生成されていません")
            if generated_ru_count != generated_ja_count:
                raise ValueError(
                    f"生成された例文の項目数が不一致（RU={generated_ru_count}, JA={generated_ja_count}）"
                )

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
    ru = get_ru_source_data(db_path, word)
    if ru is None:
        return None, "raw_data(ru_wiktionary)にデータが無いためスキップ（先にscraping.pyを実行してください）"

    source_hash = compute_source_hash(ru)

    cached = load_vocab_from_cache(db_path, word, source_hash)
    if cached is not None:
        return cached, None

    try:
        generated, raw_output = call_llm_api(word, ru, provider_cfg)
    except VocabGenerationError as e:
        return None, str(e)

    save_vocab_to_cache(
        db_path, word, source_hash, ru, generated,
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
        description="raw_data(scraping.pyの取得結果)から、外部LLM API(Groq/Gemini/OpenRouter)を使って"
                    "日本語の単語帳データを直接生成し、CSVとDB(vocab_final)に保存するツール"
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

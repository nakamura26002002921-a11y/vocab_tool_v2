-- ============================================================
-- dictionary.db 初期化スクリプト
-- ロシア語単語帳自動生成システム用のキャッシュ/中間データテーブル
-- ============================================================

PRAGMA journal_mode = WAL;   -- 並列書き込み時の競合を減らす

-- ----------------------------------------------------------
-- raw_data: scraping.py が取得した生データのキャッシュ
--   キー: (word, source_url)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_data (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    word        TEXT NOT NULL,
    source_url  TEXT NOT NULL,
    raw_html    TEXT,             -- 取得した生wikitext（デバッグ・再パース用途、任意。カラム名は互換性のため raw_html のまま）
    extracted   TEXT NOT NULL,    -- XPathで抽出したテキストをJSON文字列化したもの
    fetched_at  TEXT NOT NULL,    -- ISO8601形式のタイムスタンプ
    status      TEXT NOT NULL DEFAULT 'ok',  -- 'ok' / 'not_found' / 'error'
    UNIQUE(word, source_url)
);

CREATE INDEX IF NOT EXISTS idx_raw_data_word ON raw_data(word);

-- ----------------------------------------------------------
-- summaries: summarize.py がLLMで構造化した結果のキャッシュ
--   キー: (word, prompt_hash)
--   prompt_hash はプロンプトテンプレート＋raw_dataの内容のハッシュ値。
--   プロンプトやraw_dataが変わった場合は別キャッシュとして扱われ、
--   古いキャッシュを誤って再利用しない。
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    word            TEXT NOT NULL,
    prompt_hash     TEXT NOT NULL,
    model           TEXT NOT NULL,
    pos             TEXT,             -- 品詞
    gender          TEXT,             -- 性（名詞のみ）
    aspect          TEXT,             -- 体（動詞のみ）
    paired_verb     TEXT,             -- 完了体/不完了体のペア動詞
    meanings_ru     TEXT,             -- 意味（ロシア語原文）
    collocations_ru TEXT,             -- コロケーション（ロシア語原文）
    examples_ru     TEXT,             -- 例文（ロシア語原文）
    accent          TEXT,             -- アクセント位置情報
    raw_llm_output  TEXT,             -- LLMの生出力（デバッグ用）
    created_at      TEXT NOT NULL,
    UNIQUE(word, prompt_hash)
);

CREATE INDEX IF NOT EXISTS idx_summaries_word ON summaries(word);

-- ----------------------------------------------------------
-- vocab_final: create_vocab.py がLLM API（Groq/Gemini/OpenRouter等）で
--   生成した最終的な単語帳データのキャッシュ
--   キー: (word, source_hash)
--   source_hash は summaries+raw_data(類義語/対義語/派生語)の内容から算出。
--   原文が変わらない限り再生成しない（API呼び出し回数の節約）。
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS vocab_final (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    word             TEXT NOT NULL,
    source_hash      TEXT NOT NULL,
    pos              TEXT,
    gender           TEXT,
    aspect           TEXT,
    paired_verb      TEXT,
    accent           TEXT,
    meaning_ja       TEXT,             -- 意味（日本語解説）
    memory_tip_ja    TEXT,             -- 覚え方（語源・イメージ・記憶術）+ コロケーション（日本語）
    examples_ru      TEXT,             -- 例文（ロシア語、" / "区切り）
    examples_ja      TEXT,             -- 例文の日本語訳（" / "区切り、examples_ruと対応）
    synonyms_ru      TEXT,             -- 類義語（ロシア語、" / "区切り）
    antonyms_ru      TEXT,             -- 対義語（ロシア語、" / "区切り）
    related_words_ru TEXT,             -- 派生語（ロシア語、" / "区切り）
    provider         TEXT,             -- 生成に使ったAPIプロバイダ名（groq/gemini/openrouter）
    model            TEXT,             -- 生成に使ったモデル名
    raw_llm_output   TEXT,             -- LLMの生出力（デバッグ用）
    created_at       TEXT NOT NULL,
    UNIQUE(word, source_hash)
);

CREATE INDEX IF NOT EXISTS idx_vocab_final_word ON vocab_final(word);

-- ----------------------------------------------------------
-- run_errors: パイプライン実行時のエラーログ（DB上にも残す）
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS run_errors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    word        TEXT NOT NULL,
    stage       TEXT NOT NULL,    -- 'scraping' / 'summarize' / 'formatter'
    message     TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);

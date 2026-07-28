-- ============================================================
-- dictionary.db 初期化スクリプト
-- ロシア語単語帳自動生成システム用のキャッシュ/中間データテーブル
--
-- パイプライン: scraping.py (raw_data) -> create_vocab.py (vocab_final)
-- ============================================================

PRAGMA journal_mode = WAL;   -- 並列書き込み時の競合を減らす

-- ----------------------------------------------------------
-- raw_data: scraping.py が取得した生データのキャッシュ
--   キー: (word, source_url)
--   create_vocab.py はこのテーブル（主に ru_wiktionary ソース）を直接読み、
--   外部LLM APIで日本語の単語帳データを生成する。
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_data (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    word        TEXT NOT NULL,
    source_url  TEXT NOT NULL,
    raw_html    TEXT,             -- 取得した生wikitext（デバッグ・再パース用途、任意。カラム名は互換性のため raw_html のまま）
    extracted   TEXT NOT NULL,    -- XPath/正規表現で抽出したテキストをJSON文字列化したもの
    fetched_at  TEXT NOT NULL,    -- ISO8601形式のタイムスタンプ
    status      TEXT NOT NULL DEFAULT 'ok',  -- 'ok' / 'not_found' / 'error'
    UNIQUE(word, source_url)
);

CREATE INDEX IF NOT EXISTS idx_raw_data_word ON raw_data(word);

-- ----------------------------------------------------------
-- vocab_final: create_vocab.py がLLM API（Groq/Gemini/OpenRouter等）で
--   raw_data の内容から直接生成した最終的な単語帳データのキャッシュ
--   キー: (word, source_hash)
--   source_hash は raw_data（ru_wiktionaryの意味/例文/品詞情報/類義語/対義語/派生語等）
--   の内容から算出する。原文が変わらない（再スクレイピングされない）限り再生成しない
--   （API呼び出し回数の節約）。
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS vocab_final (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    word             TEXT NOT NULL,
    source_hash      TEXT NOT NULL,
    pos              TEXT,             -- 品詞（LLMがpos_blockから抽出）
    gender           TEXT,             -- 性（名詞のみ、LLMがpos_blockから抽出）
    aspect           TEXT,             -- 体（動詞のみ、LLMがpos_blockから抽出）
    paired_verb      TEXT,             -- 完了体/不完了体のペア動詞（LLMがpos_blockから抽出）
    accent           TEXT,             -- アクセント情報（raw_dataのaccent_wordより、決定的に取得）
    meaning_ja       TEXT,             -- 意味（日本語解説）
    memory_tip_ja    TEXT,             -- 覚え方（語源・イメージ・記憶術）+ コロケーション（日本語）
    examples_ru      TEXT,             -- 例文（ロシア語、raw_dataより決定的に取得、" / "区切り）
    examples_ja      TEXT,             -- 例文の日本語訳（LLM生成、" / "区切り、examples_ruと対応）
    synonyms_ru      TEXT,             -- 類義語（ロシア語、raw_dataより決定的に取得、" / "区切り）
    antonyms_ru      TEXT,             -- 対義語（ロシア語、raw_dataより決定的に取得、" / "区切り）
    related_words_ru TEXT,             -- 派生語（ロシア語、raw_dataより決定的に取得、" / "区切り）
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
    stage       TEXT NOT NULL,    -- 'scraping' / 'create_vocab'
    message     TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);

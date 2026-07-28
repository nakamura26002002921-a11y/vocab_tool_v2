"""
common.py
---------
全モジュール（scraping.py / summarize.py / formatter.py / main.py）で共有する
設定読み込み・DB接続・ロギングのユーティリティ。
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

CONFIG_PATH = "config.json"
INIT_SQL_PATH = "init_db.sql"

# スレッドごとに一度だけ「DBが初期化済みか」を確認するためのロック
_db_init_lock = threading.Lock()
_db_initialized = False


# ---------------------------------------------------------------------------
# 設定読み込み
# ---------------------------------------------------------------------------
def _resolve_env_placeholder(raw: str) -> str:
    """"${VAR:-default}" 形式の環境変数参照を解決する"""
    if isinstance(raw, str) and raw.startswith("${") and raw.endswith("}") and ":-" in raw:
        inner = raw[2:-1]
        var_name, default_val = inner.split(":-", 1)
        return os.environ.get(var_name, default_val)
    return raw


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 環境変数プレースホルダの解決
    if "llm" in config and "base_url" in config["llm"]:
        config["llm"]["base_url"] = _resolve_env_placeholder(config["llm"]["base_url"]).rstrip("/")

    if "polish_llm" in config and "base_url" in config["polish_llm"]:
        config["polish_llm"]["base_url"] = _resolve_env_placeholder(config["polish_llm"]["base_url"]).rstrip("/")

    # デフォルト値の補完
    config.setdefault("database", {}).setdefault("path", "dictionary.db")
    llm = config.setdefault("llm", {})
    llm.setdefault("temperature", 0.0)
    llm.setdefault("max_tokens", 1024)
    llm.setdefault("timeout_seconds", 120)
    llm.setdefault("max_retries", 2)

    scraping = config.setdefault("scraping", {})
    scraping.setdefault("sources", [])
    scraping.setdefault("user_agent", "Mozilla/5.0 (compatible; VocabToolBot/1.0)")

    pipeline = config.setdefault("pipeline", {})
    pipeline.setdefault("max_workers", 4)
    pipeline.setdefault("on_error", "keep_as_error_row")
    pipeline.setdefault("input_file", "words.txt")
    pipeline.setdefault("output_file", "vocab.csv")
    pipeline.setdefault("csv_bom", True)
    pipeline.setdefault("log_file", "logs/errors.log")

    vocab_llm = config.setdefault("vocab_llm", {})
    vocab_llm.setdefault("provider", "groq")
    for provider_cfg in vocab_llm.setdefault("providers", {}).values():
        provider_cfg.setdefault("temperature", 0.2)
        provider_cfg.setdefault("max_tokens", 2048)
        provider_cfg.setdefault("timeout_seconds", 60)
        provider_cfg.setdefault("max_retries", 3)
        provider_cfg.setdefault("request_delay_seconds", 0.5)

    return config


# ---------------------------------------------------------------------------
# DB接続
# ---------------------------------------------------------------------------
def ensure_db_initialized(db_path: str, init_sql_path: str = INIT_SQL_PATH) -> None:
    """dictionary.db にテーブルが存在しなければ init_db.sql を実行して作成する。
    スレッドセーフに一度だけ実行する。"""
    global _db_initialized
    with _db_init_lock:
        if _db_initialized:
            return
        conn = sqlite3.connect(db_path)
        try:
            with open(init_sql_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())
            conn.commit()
        finally:
            conn.close()
        _db_initialized = True


@contextmanager
def get_connection(db_path: str):
    """呼び出しのたびに新しい接続を作る（スレッドごとに独立した接続を持つのが
    SQLiteでの安全な並列アクセス方法のため、コネクションプールは使わない）。"""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# ロギング
# ---------------------------------------------------------------------------
def setup_logger(log_file: str, name: str = "vocab_tool") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 既に設定済み（複数モジュールから呼ばれても二重登録しない）

    logger.setLevel(logging.INFO)

    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def record_error(db_path: str, word: str, stage: str, message: str) -> None:
    """run_errors テーブルにエラーを記録する。"""
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO run_errors (word, stage, message, occurred_at) VALUES (?, ?, ?, ?)",
            (word, stage, message, now_iso()),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# 汎用テキスト処理
# ---------------------------------------------------------------------------
def strip_code_fences(text: str) -> str:
    text = re.sub(r"^```(?:csv|text|json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


def cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(CYRILLIC_RE.findall(text)) / max(len(text), 1)

# vocab_tool_v2

ロシア語単語帳（日本人のロシア語学習者向け）を自動生成するツール群です。
ロシア語版Wiktionary等からデータをスクレイピングし、無料枠の外部LLM API
（Groq / Gemini / OpenRouter）を1回呼び出すだけで日本語の単語帳データを生成、
CSVファイルとして出力します。

## パイプライン全体像

```
words.txt (見出し語リスト)
   │
   ▼
① scraping.py ──────► raw_data テーブル
   （ru.wiktionary.org 等から意味・例文・品詞情報・類義語/対義語/派生語等を取得）
   │
   ▼
② create_vocab.py ──► vocab_final テーブル ＋ CSV出力
   （raw_data を直接読み、外部LLM API(Groq/Gemini/OpenRouter)で
     文法情報の抽出 と 日本語の単語帳カード生成 を1回のAPI呼び出しで行う）
```

**ローカルLLM（Ollama）を使った中間の構造化ステージはありません。** 以前のバージョンでは
`summarize.py` でロシア語データをいったんローカルLLMで構造化していましたが、本バージョンでは
廃止し、`scraping.py` の生データを `create_vocab.py` が直接読み込みます。文法情報（品詞・性・体・
ペア動詞）の判定と日本語解説の生成は、Groq/Gemini等の外部APIモデルにまとめて行わせます。

各ステージの結果はすべて `dictionary.db`（SQLite）にキャッシュされます。
同じ入力（`raw_data`の内容）に対しては再実行してもAPI呼び出しを行わず、キャッシュ
（`vocab_final`テーブル）から読み出すだけなので、`--startidx` / `--endidx` を変えながら
何度でも安全に再実行できます。

## ハルシネーション対策

- 例文原文・アクセント・類義語・対義語・派生語など「事実」に関わる情報は `raw_data` から
  そのまま決定的に転記し、LLMには生成させません。
- LLMに生成させるのは、日本語での意味解説・覚え方・例文訳という「言語表現」部分のみです。
- 例文の日本語訳は、ロシア語原文と同じ数・同じ順序の `" / "` 区切り項目になるようプロンプトで
  指示し、項目数が一致しない場合はその単語の生成を失敗として扱います（対応関係が崩れたデータを
  CSV/DBに混入させないためのフェイルセーフ）。

## セットアップ

### 1. Python環境

Python 3.10以上を推奨します。

```bash
pip install -r requirements.txt
```

### 2. 外部LLM APIキー

`config.json` の `vocab_llm.providers` に、使用したいプロバイダのAPIキーを設定してください。
`provider` で使用するプロバイダを切り替えます（デフォルトは `groq`）。

```json
"vocab_llm": {
  "provider": "groq",
  "providers": {
    "groq":       { "api_key": "YOUR_GROQ_API_KEY", ... },
    "gemini":     { "api_key": "YOUR_GEMINI_API_KEY", ... },
    "openrouter": { "api_key": "YOUR_OPENROUTER_API_KEY", ... }
  }
}
```

- Groq: https://console.groq.com/ （無料枠あり）
- Google AI Studio (Gemini): https://aistudio.google.com/ （無料枠あり）
- OpenRouter: https://openrouter.ai/ （無料モデルあり）

`api_key` が `"YOUR_..."` のままだとエラーになるので、必ず実際のキーに書き換えてください。
実行時に `--provider` オプションで一時的に切り替えることもできます。

```bash
python3 create_vocab.py --provider gemini --startidx 1 --endidx 10
```

### 3. 単語リスト

`words.txt` に1行1単語（キリル文字）でロシア語の見出し語を並べます。同梱の `words.txt` は
約5万語収録されています。

## 使い方

範囲を指定しながら2ステージを順番に実行します（`--startidx` / `--endidx` は1始まりの行番号）。

```bash
# ① スクレイピング（raw_data に保存）
python3 scraping.py --startidx 1 --endidx 100

# ② 外部LLM APIで日本語の単語帳を直接生成し、CSVに出力
python3 create_vocab.py --startidx 1 --endidx 100 --output vocab.csv
```

### 主なオプション

| スクリプト | オプション | 説明 |
|---|---|---|
| 共通 | `--startidx` / `--endidx` | 処理する行範囲（1始まり、`words.txt`基準） |
| 共通 | `--input` | 入力ファイル（デフォルト: `words.txt`） |
| `scraping.py` | `--force` | 既存の `raw_data` キャッシュを無視して再取得 |
| `scraping.py` | `--quiet` | 抽出結果JSONを標準出力に表示しない |
| `create_vocab.py` | `--output` | 出力CSVファイル名（デフォルト: `config.json`の`pipeline.output_file`） |
| `create_vocab.py` | `--provider` | 使用するAPIプロバイダを一時的に上書き（`groq`/`gemini`/`openrouter`） |

## 出力CSVのカラム

`create_vocab.py` が出力するCSV（デフォルト `vocab.csv`, UTF-8 with BOM）は以下の列を持ちます。

| 列名 | 内容 | 生成方法 |
|---|---|---|
| Word | 見出し語（ロシア語） | 入力そのまま |
| Accent | アクセント位置情報 | raw_data から決定的に転記 |
| POS | 品詞 | LLMが文法情報の自然文から判定 |
| Gender | 性（名詞のみ） | LLMが文法情報の自然文から判定 |
| Aspect | 体（動詞のみ） | LLMが文法情報の自然文から判定 |
| PairedVerb | 完了体/不完了体のペア動詞 | LLMが文法情報の自然文から判定 |
| Meaning_JA | 意味（日本語解説、複数ある場合は" / "区切り） | LLM生成 |
| MemoryTip_JA | 覚え方・語源・代表的なコロケーション（日本語） | LLM生成 |
| Examples_RU | 例文（ロシア語、" / "区切り） | raw_data から決定的に転記 |
| Examples_JA | 例文の日本語訳（Examples_RUと同じ数・同じ順序） | LLM生成（項目数不一致時は失敗扱い） |
| Synonyms_RU | 類義語（ロシア語） | raw_data から決定的に転記 |
| Antonyms_RU | 対義語（ロシア語） | raw_data から決定的に転記 |
| RelatedWords_RU | 派生語・関連語（ロシア語） | raw_data から決定的に転記 |

## データの中身（DBテーブル）

`dictionary.db`（SQLite）に以下のテーブルが作られます。詳細は `init_db.sql` を参照してください。

- `raw_data` — scraping.pyが取得した生データ（`word` + `source_url` でキャッシュ）
- `vocab_final` — create_vocab.pyが生成した日本語単語帳データ（`word` + `source_hash` でキャッシュ）
- `run_errors` — 各ステージのエラーログ

## トラブルシューティング

- **`vocab_llm.provider=... が vocab_llm.providers に定義されていません` と出る**
  `config.json` の `vocab_llm.provider` の値と `vocab_llm.providers` のキー名が一致しているか確認してください。
- **`api_key が未設定です` と出る**
  `config.json` の該当プロバイダの `api_key` を実際のキーに書き換えてください。
- **`raw_data(ru_wiktionary)にデータが無いためスキップ` と出る**
  対象の単語がまだ①（`scraping.py`）で処理されていないか、Wiktionaryにページが存在しません。
  先に①を実行するか、その単語を諦めてください。
- **`例文の項目数が不一致` でスキップされる**
  LLMが例文訳の対応関係を崩した場合のフェイルセーフです。プロバイダやモデルを変えて
  再実行すると解消することがあります（`--provider`）。
- スクレイピング先サイト（multitran, reverso_context等）はHTML構造の変更に弱い正規表現ベースの
  抽出です。取得結果が空になる場合は `config.json` の `scraping.sources[].field_map` の
  正規表現を実際のHTMLに合わせて調整してください（これらのソースは補助的な英訳参考情報として
  プロンプトに渡されるのみで、必須ではありません）。

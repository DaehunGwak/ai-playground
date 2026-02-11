# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PDF 교재를 마크다운으로 변환하고, 로컬 Ollama 임베딩을 거쳐 Milvus 벡터 DB에 저장한 뒤, LangGraph 기반 RAG로 질의응답하는 파이프라인.

## Tech Stack

- **Python 3.13** / **uv** (패키지 매니저)
- **Ollama**: 로컬 LLM(`qwen3:14b`) 및 임베딩(`qwen3-embedding:8b`, dim=4096)
- **Milvus v2.6.10** (standalone, Docker Compose): 벡터 DB
- **LangGraph** + **LangChain**: RAG 파이프라인 오케스트레이션
- **Streamlit**: 챗봇 웹 UI

## Commands

```bash
# 의존성
uv sync
uv add <package>

# PDF → Markdown 변환
uv run scripts/pdf_to_md.py resources/파일.pdf -o output/파일.md
uv run scripts/pdf_to_md.py resources/파일.pdf --pages 10-50

# Markdown → Milvus 임베딩 저장 (중단 후 재실행 시 이어서 처리)
uv run scripts/embed_to_milvus.py output/파일.md --collection music_processing_book

# CLI RAG 검색
uv run scripts/rag_search.py "검색 질문" --top-k 5 --chapter "Chapter 2"

# Streamlit 챗봇
uv run streamlit run app/chat.py

# Milvus 시작/종료
cd milvus && docker compose up -d
cd milvus && docker compose down
```

## Architecture

파이프라인: `PDF → Markdown → 청킹 → 임베딩 → Milvus → LangGraph RAG → 응답`

- **`scripts/pdf_to_md.py`** — pymupdf4llm으로 PDF를 마크다운 변환. 페이지 범위 지정 가능.
- **`scripts/embed_to_milvus.py`** — 마크다운을 헤딩 기준 청킹 → Ollama 배치 임베딩 → Milvus 저장. 배치마다 즉시 저장하여 중단 시 재개 가능. 기존 chunk_index를 조회해 중복 임베딩 방지.
- **`scripts/rag_search.py`** — LangGraph 기반 CLI RAG. `retrieve`(Milvus 벡터 검색) → `generate`(LLM 응답) 2단계 그래프.
- **`app/chat.py`** — Streamlit 채팅 UI. rag_search.py와 동일한 LangGraph 그래프를 사용. 사이드바에서 top-k, 챕터 필터 설정.
- **`milvus/docker-compose.yml`** — Milvus standalone (etcd + MinIO + Milvus + Attu).

## Milvus Collection Schema (`music_processing_book`)

| Field | Type | Index |
|-------|------|-------|
| id | INT64 (PK) | auto |
| vector | FLOAT_VECTOR(4096) | AUTOINDEX, COSINE |
| text | VARCHAR(65535) | none |
| heading | VARCHAR(512) | none |
| chapter | VARCHAR(128) | INVERTED |
| source | VARCHAR(256) | none |
| chunk_index | INT32 | INVERTED |

## Key Ports

| Service | Port |
|---------|------|
| Milvus gRPC | 19530 |
| Milvus WebUI | 9091 |
| MinIO API / Console | 9000 / 9001 |
| Attu (Milvus UI) | 8000 |
| Streamlit | 8501 |

## Conventions

- 모든 스크립트는 `uv run`으로 실행
- `resources/`, `output/`, `milvus/volumes/`는 .gitignore 대상
- 스크립트 내 주석과 출력 메시지는 한국어
- Ollama 모델은 로컬에서 직접 pull (`ollama pull qwen3-embedding:8b`)

# CLAUDE.md

## Project Overview

LLM을 밑바닥부터 구현하며 학습하는 프로젝트. Python 3.13 + PyTorch + Jupyter 환경.

## Tech Stack

- Python 3.13
- PyTorch
- Jupyter Notebook / JupyterLab
- 패키지 매니저: uv

## Commands

- `uv run python <script>` — Python 스크립트 실행
- `uv run jupyter notebook` — Jupyter Notebook 실행
- `uv run jupyter lab` — JupyterLab 실행
- `uv add <package>` — 의존성 추가

## Project Structure

```
llm-scratch/
├── main.py          # 엔트리포인트
├── notebooks/       # Jupyter 노트북
├── pyproject.toml   # 프로젝트 설정 및 의존성
├── uv.lock          # 의존성 잠금 파일
└── .python-version  # Python 버전 (3.13)
```

## Conventions

- 언어: 코드 주석 및 노트북 설명은 한국어 사용
- 모든 명령은 `uv run`을 통해 실행
- 새로운 패키지 추가 시 `uv add` 사용

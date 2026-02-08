# LLM Scratch

'밑바닥부터 만들면서 배우는 LLM' 책을 따라가며 정리한 실습 코드입니다.

## 환경 설정

- Python 3.13
- PyTorch
- NumPy
- Jupyter (dev)

### 설치

```bash
# 전체 설치 (Jupyter 포함)
uv sync

# 핵심 패키지만 설치
uv sync --no-dev
```

### 실행

```bash
# Jupyter Notebook
uv run jupyter notebook

# JupyterLab
uv run jupyter lab

# Python 스크립트
uv run python main.py
```

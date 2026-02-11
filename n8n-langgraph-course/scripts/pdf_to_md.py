"""PDF to Markdown 변환 스크립트

사용법:
    uv run pdf_to_md.py resources/교재.pdf                    # 같은 위치에 .md 생성
    uv run pdf_to_md.py resources/교재.pdf -o output/교재.md   # 출력 경로 지정
    uv run pdf_to_md.py resources/교재.pdf --pages 10-50       # 특정 페이지만 변환
"""

import argparse
import pathlib

import pymupdf4llm


def parse_pages(pages_str: str) -> list[int]:
    """'10-50' 같은 페이지 범위를 리스트로 변환 (0-indexed)"""
    start, end = pages_str.split("-")
    return list(range(int(start) - 1, int(end)))


def main():
    parser = argparse.ArgumentParser(description="PDF to Markdown 변환")
    parser.add_argument("pdf", type=pathlib.Path, help="변환할 PDF 파일 경로")
    parser.add_argument("-o", "--output", type=pathlib.Path, help="출력 Markdown 파일 경로")
    parser.add_argument("--pages", type=str, help="변환할 페이지 범위 (예: 10-50)")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"파일을 찾을 수 없습니다: {args.pdf}")
        return

    output = args.output or args.pdf.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    if args.pages:
        kwargs["pages"] = parse_pages(args.pages)

    print(f"변환 중: {args.pdf}")
    md = pymupdf4llm.to_markdown(str(args.pdf), **kwargs)

    output.write_text(md, encoding="utf-8")
    print(f"완료: {output}")


if __name__ == "__main__":
    main()

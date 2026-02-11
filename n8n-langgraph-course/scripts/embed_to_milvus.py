"""Markdown 파일을 청킹 → Ollama 임베딩 → Milvus 저장 스크립트

사용법:
    uv run scripts/embed_to_milvus.py output/파일.md
    uv run scripts/embed_to_milvus.py output/파일.md --collection my_docs
    uv run scripts/embed_to_milvus.py output/파일.md --chunk-size 1000 --overlap 200
"""

import argparse
import pathlib
import re
import time

import ollama
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

EMBEDDING_MODEL = "qwen3-embedding:8b"
EMBEDDING_DIM = 4096
MILVUS_URI = "http://localhost:19530"

# 챕터 패턴: "Chapter 1", "Chapter 2" 등
CHAPTER_PATTERN = re.compile(r"\*{0,2}Chapter\s+(\d+)\*{0,2}")


def detect_chapter(heading: str, prev_chapter: str) -> str:
    """헤딩에서 챕터 정보를 추출, 없으면 이전 챕터 유지"""
    match = CHAPTER_PATTERN.search(heading)
    if match:
        return f"Chapter {match.group(1)}"
    return prev_chapter


def split_by_heading(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[dict]:
    """마크다운 헤딩 기준으로 분할 후, 긴 섹션은 chunk_size로 재분할"""
    sections = re.split(r"\n(?=#{1,6}\s)", text)
    chunks = []
    current_chapter = "Front Matter"

    for section in sections:
        section = section.strip()
        if not section:
            continue

        heading_match = re.match(r"(#{1,6})\s+(.*)", section)
        heading = heading_match.group(2).strip() if heading_match else ""

        current_chapter = detect_chapter(heading, current_chapter)

        if len(section) <= chunk_size:
            chunks.append({"heading": heading, "chapter": current_chapter, "text": section})
            continue

        paragraphs = section.split("\n\n")
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > chunk_size and current:
                chunks.append({"heading": heading, "chapter": current_chapter, "text": current.strip()})
                words = current.split()
                overlap_text = " ".join(words[-overlap // 5 :]) if len(words) > overlap // 5 else ""
                current = overlap_text + "\n\n" + para
            else:
                current = current + "\n\n" + para if current else para
        if current.strip():
            chunks.append({"heading": heading, "chapter": current_chapter, "text": current.strip()})

    return chunks


def clean_text(text: str) -> str:
    """텍스트에서 의미 없는 문자를 제거하고 알파벳/숫자 비율 확인용 정제"""
    return re.sub(r"[#*_\-\s\n\r\t|>]+", "", text)


def filter_chunks(chunks: list[dict], min_length: int = 30) -> list[dict]:
    """의미 있는 텍스트가 부족한 청크 필터링"""
    filtered = []
    skipped = 0
    for chunk in chunks:
        cleaned = clean_text(chunk["text"])
        if len(cleaned) >= min_length:
            filtered.append(chunk)
        else:
            skipped += 1
    if skipped:
        print(f"  {skipped}개 청크 필터링됨 (텍스트 부족)")
    return filtered


def embed_and_insert_batch(
    client: MilvusClient,
    collection: str,
    chunks: list[tuple[int, dict]],
    source: str,
    batch_size: int = 16,
) -> tuple[int, int]:
    """배치 임베딩 후 즉시 Milvus에 저장. (저장 수, 스킵 수) 반환"""
    saved, skipped = 0, 0
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        texts = [c["text"] for _, c in batch_chunks]
        indexes = [idx for idx, _ in batch_chunks]
        embeddings = []

        try:
            response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
            embeddings = list(zip(indexes, batch_chunks, response.embeddings))
        except Exception:
            # 배치 실패 시 개별 처리
            for idx, chunk_pair in zip(indexes, batch_chunks):
                try:
                    response = ollama.embed(model=EMBEDDING_MODEL, input=[chunk_pair[1]["text"]])
                    embeddings.append((idx, chunk_pair, response.embeddings[0]))
                except Exception:
                    print(f"  [SKIP] 청크 {idx} 임베딩 실패: {chunk_pair[1]['text'][:60]}...")
                    skipped += 1

        if embeddings:
            data = [
                {
                    "id": idx,
                    "vector": emb,
                    "text": cp[1]["text"],
                    "heading": cp[1]["heading"],
                    "chapter": cp[1]["chapter"],
                    "source": source,
                    "chunk_index": idx,
                }
                for idx, cp, emb in embeddings
            ]
            client.insert(collection_name=collection, data=data)
            saved += len(data)

        print(f"  진행: {min(i + batch_size, total)}/{total} (저장: {saved}, 스킵: {skipped})")

    return saved, skipped


def ensure_collection(client: MilvusClient, name: str):
    """컬렉션이 없으면 생성, 있으면 기존 컬렉션 사용"""
    if client.has_collection(name):
        print(f"  기존 컬렉션 '{name}' 사용")
        return

    schema = CollectionSchema(description="Music Processing 교재 임베딩")
    schema.add_field(FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True))
    schema.add_field(FieldSchema("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM))
    schema.add_field(FieldSchema("text", DataType.VARCHAR, max_length=65535))
    schema.add_field(FieldSchema("heading", DataType.VARCHAR, max_length=512))
    schema.add_field(FieldSchema("chapter", DataType.VARCHAR, max_length=128))
    schema.add_field(FieldSchema("source", DataType.VARCHAR, max_length=256))
    schema.add_field(FieldSchema("chunk_index", DataType.INT32))

    client.create_collection(collection_name=name, schema=schema)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", metric_type="COSINE", index_type="AUTOINDEX")
    index_params.add_index(field_name="chapter", index_type="INVERTED")
    index_params.add_index(field_name="chunk_index", index_type="INVERTED")
    client.create_index(collection_name=name, index_params=index_params)

    print(f"  컬렉션 '{name}' 생성 완료 (vector: COSINE/AUTOINDEX, chapter+chunk_index: INVERTED)")


def get_existing_chunk_indexes(client: MilvusClient, collection: str, source: str) -> set[int]:
    """컬렉션에서 이미 저장된 chunk_index 목록 조회 (페이지네이션)"""
    try:
        client.load_collection(collection)
        existing = set()
        offset = 0
        page_size = 1000
        while True:
            results = client.query(
                collection_name=collection,
                filter=f'source == "{source}"',
                output_fields=["chunk_index"],
                limit=page_size,
                offset=offset,
            )
            if not results:
                break
            existing.update(r["chunk_index"] for r in results)
            offset += page_size
        return existing
    except Exception:
        return set()


def main():
    parser = argparse.ArgumentParser(description="Markdown → Milvus 임베딩 저장")
    parser.add_argument("markdown", type=pathlib.Path, help="입력 Markdown 파일")
    parser.add_argument("--collection", type=str, default="music_processing", help="Milvus 컬렉션 이름")
    parser.add_argument("--chunk-size", type=int, default=1500, help="청크 최대 글자 수")
    parser.add_argument("--overlap", type=int, default=200, help="청크 간 겹침 글자 수")
    args = parser.parse_args()

    if not args.markdown.exists():
        print(f"파일을 찾을 수 없습니다: {args.markdown}")
        return

    # 0. Milvus 연결 & 컬렉션 확인
    client = MilvusClient(uri=MILVUS_URI)
    ensure_collection(client, args.collection)

    # 1. 마크다운 읽기 & 청킹
    print(f"[1/4] 마크다운 읽기 & 청킹: {args.markdown}")
    text = args.markdown.read_text(encoding="utf-8")
    chunks = split_by_heading(text, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"  총 {len(chunks)}개 청크 생성")
    chunks = filter_chunks(chunks)
    print(f"  필터링 후 {len(chunks)}개 청크")

    # 2. 이미 저장된 청크 확인
    print(f"[2/4] 기존 임베딩 확인")
    source_name = args.markdown.name
    existing = get_existing_chunk_indexes(client, args.collection, source_name)
    new_chunks = [(i, c) for i, c in enumerate(chunks) if i not in existing]
    print(f"  기존: {len(existing)}개, 신규: {len(new_chunks)}개")

    if not new_chunks:
        print("\n모든 청크가 이미 저장되어 있습니다. 완료!")
        return

    # 3. 임베딩 + 즉시 저장
    print(f"[3/3] 임베딩 & 저장 (모델: {EMBEDDING_MODEL})")
    start = time.time()
    saved, skipped = embed_and_insert_batch(client, args.collection, new_chunks, source_name)
    elapsed = time.time() - start
    print(f"  완료: {elapsed:.1f}초 (저장: {saved}, 스킵: {skipped})")

    client.load_collection(args.collection)
    stats = client.get_collection_stats(args.collection)
    print(f"\n완료! 컬렉션 '{args.collection}': {stats['row_count']}개 문서")


if __name__ == "__main__":
    main()

"""
bench.py — Script đánh giá Benchmark RAG trên bộ tài liệu học bổng.

Thực hiện:
1. Nạp bộ 5 câu hỏi và đáp án chuẩn (Gold Set) từ file JSON riêng biệt.
2. Đọc các file .md trong data/scholarship/, tách frontmatter metadata và body content.
3. Áp dụng chiến lược chunking được chọn để chia nhỏ nội dung thành các Document chunks.
4. Nạp tất cả chunk vào EmbeddingStore.
5. Chạy các câu hỏi benchmark, truy xuất top-3 kết quả (kèm metadata filter cho câu cần lọc).
6. Đánh giá 2 mức (Mức 1: Trúng Gold Doc, Mức 2: Chunk chứa từ khóa đáp án thực tế).
7. In kết quả ra màn hình và đồng thời lưu vào ket_qua_benchmark.txt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import (
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
)
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore


# =====================================================================
# 1. Custom Chunker: Markdown Heading / Section Chunker
# =====================================================================
class MarkdownHeadingChunker:
    """
    Chiến lược chunking nâng cao dựa trên cấu trúc đề mục Markdown / Điều khoản.
    Nếu một section vượt quá max_chunk_size, phân rã bằng RecursiveChunker
    và gắn lại tiêu đề của section vào từng mảnh con để giữ trọn ngữ cảnh.
    """

    def __init__(self, max_chunk_size: int = 400) -> None:
        self.max_chunk_size = max_chunk_size
        self.recursive_fallback = RecursiveChunker(chunk_size=max_chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        heading_pattern = re.compile(
            r"(?=(?:^|\n)(?:#{1,4}\s+|Điều\s+\d+[\.:]|(?:[1-9]\d?|\b[IVXLCDM]+\b)\.\s+))",
            re.M,
        )
        raw_sections = heading_pattern.split(text)
        sections = [s.strip() for s in raw_sections if s.strip()]

        chunks: list[str] = []
        for sec in sections:
            if len(sec) <= self.max_chunk_size:
                chunks.append(sec)
            else:
                lines = sec.split("\n", 1)
                title = lines[0].strip()
                body = lines[1].strip() if len(lines) > 1 else ""
                if not body:
                    chunks.append(title)
                    continue

                sub_chunks = self.recursive_fallback.chunk(body)
                for sub in sub_chunks:
                    chunks.append(f"{title}\n{sub}")

        return chunks


# =====================================================================
# 2. Nạp Gold Set từ file riêng biệt
# =====================================================================
def load_benchmark_queries(queries_path: Path | str | None = None) -> list[dict]:
    """
    Tải danh sách benchmark query kèm gold answers từ file JSON riêng biệt.
    Ưu tiên đường dẫn người dùng chỉ định, sau đó thử các vị trí chuẩn trong repo.
    """
    candidates = []
    if queries_path:
        candidates.append(Path(queries_path))
    candidates.extend([
        Path("data/scholarship/benchmark_gold_set.json"),
        Path("benchmark_gold_set.json"),
    ])

    for path in candidates:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data
            except Exception as err:
                print(f"[!] Cảnh báo: Lỗi khi đọc file gold set '{path}': {err}", file=sys.stderr)

    raise FileNotFoundError(
        "Không tìm thấy file benchmark gold set! "
        "Vui lòng tạo file 'data/scholarship/benchmark_gold_set.json' hoặc truyền đường dẫn qua --queries-file."
    )


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Tách frontmatter và nội dung phần thân của file markdown."""
    parts = text.split("---", 2)
    if len(parts) >= 3:
        front_text = parts[1]
        body = parts[2].strip()
        metadata = {}
        for line in front_text.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                metadata[k.strip()] = v.strip().strip('"\'')
        return metadata, body
    return {}, text.strip()


def get_embedder():
    """Lấy embedding model cấu hình trong môi trường hoặc fallback sang mock."""
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    elif provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    elif provider == "gemini":
        try:
            return GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    return _mock_embed


def run_benchmark(
    strategy_name: str = "heading",
    data_dir: str = "data/scholarship",
    queries_file: str | None = None,
) -> str:
    # -------------------------------------------------------------
    # BƯỚC 1: Chọn chiến lược chunking (chỉ cần đổi dòng này)
    # Các lựa chọn: "heading", "recursive", "fixed", "sentences"
    # -------------------------------------------------------------
    if strategy_name == "heading":
        chunker = MarkdownHeadingChunker(max_chunk_size=400)
    elif strategy_name == "recursive":
        chunker = RecursiveChunker(chunk_size=300)
    elif strategy_name == "fixed":
        chunker = FixedSizeChunker(chunk_size=300, overlap=50)
    elif strategy_name == "sentences":
        chunker = SentenceChunker(max_sentences_per_chunk=3)
    else:
        raise ValueError(f"Chiến lược không hợp lệ: {strategy_name}")

    # Nạp bộ benchmark từ file riêng
    benchmark_queries = load_benchmark_queries(queries_file)

    embedder = get_embedder()
    store = EmbeddingStore(collection_name=f"benchmark_{strategy_name}", embedding_fn=embedder)

    dir_path = Path(data_dir)
    md_files = sorted(dir_path.glob("*.md"))

    output_lines = []
    output_lines.append(f"=================================================================")
    output_lines.append(f"           KẾT QUẢ BENCHMARK RETRIEVAL — LAB 07")
    output_lines.append(f" Chiến lược: {strategy_name.upper()} ({chunker.__class__.__name__})")
    output_lines.append(f" Thư mục dữ liệu: {data_dir} ({len(md_files)} files .md)")
    output_lines.append(f" Số câu hỏi benchmark: {len(benchmark_queries)} câu (load từ JSON)")
    output_lines.append(f" Embedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")
    output_lines.append(f"=================================================================\n")

    # -------------------------------------------------------------
    # BƯỚC 2: Đọc file, chunking bên ngoài store và nạp vào Document
    # -------------------------------------------------------------
    all_docs: list[Document] = []
    for p in md_files:
        text = p.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        chunks = chunker.chunk(body)

        for i, ch in enumerate(chunks):
            chunk_doc = Document(
                id=f"{p.stem}#{i}",
                content=ch,
                metadata={**meta, "doc_id": p.stem, "chunk_index": i},
            )
            all_docs.append(chunk_doc)

    store.add_documents(all_docs)
    output_lines.append(f"[+] Đã chia nhỏ và nạp tổng cộng {store.get_collection_size()} chunks vào EmbeddingStore.\n")

    # -------------------------------------------------------------
    # BƯỚC 3: Đánh giá từng câu hỏi trong Gold Set
    # -------------------------------------------------------------
    total_score = 0
    total_content_hits = 0

    for item in benchmark_queries:
        q_id = item["id"]
        query = item["query"]
        gold_id = item["gold_doc_id"]
        gold_ans = item["gold_answer"]
        f_meta = item.get("filter")
        expected_kws = item.get("expected_keywords", [])

        output_lines.append("-" * 65)
        output_lines.append(f"CÂU HỎI {q_id}: {query}")
        output_lines.append(f"  • Filter: {f_meta}")
        output_lines.append(f"  • Gold Doc: {gold_id}")
        output_lines.append(f"  • Gold Answer: {gold_ans}")
        if expected_kws:
            output_lines.append(f"  • Từ khóa kiểm tra đáp án: {expected_kws}")

        results = store.search_with_filter(query, top_k=3, metadata_filter=f_meta)

        matched_doc_rank = None
        matched_content_rank = None

        for rank, res in enumerate(results, start=1):
            res_doc_id = res["metadata"].get("doc_id", "")
            is_gold_doc = (res_doc_id == gold_id)
            has_keywords = any(kw.lower() in res["content"].lower() for kw in expected_kws) if expected_kws else False

            if is_gold_doc and matched_doc_rank is None:
                matched_doc_rank = rank
            if is_gold_doc and has_keywords and matched_content_rank is None:
                matched_content_rank = rank

            tags = []
            if is_gold_doc:
                tags.append("TRÚNG GOLD DOC")
            if has_keywords:
                tags.append("CHỨA TỪ KHÓA ĐÁP ÁN")

            marker = f"===> [{' | '.join(tags)}]" if tags else ""
            preview = res["content"][:120].replace("\n", " ")
            output_lines.append(f"  [{rank}] Score: {res['score']:.4f} | doc_id: {res_doc_id} {marker}")
            output_lines.append(f"      Snippet: {preview}...")

        # Chấm điểm 2 mức theo hướng dẫn Phụ lục Lab:
        # Mức 1: Gold doc xuất hiện ở đâu trong Top-3
        if matched_doc_rank == 1:
            item_score = 2
        elif matched_doc_rank in (2, 3):
            item_score = 1
        else:
            item_score = 0

        if matched_content_rank is not None:
            total_content_hits += 1

        total_score += item_score
        output_lines.append(f"  ==> Đánh giá: {item_score}/2 điểm (Gold Doc: Top-{matched_doc_rank if matched_doc_rank else 'None'} | Chứa đáp án thực tế: {'CÓ (Top-' + str(matched_content_rank) + ')' if matched_content_rank else 'KHÔNG'})")
        output_lines.append("")

    output_lines.append("=" * 65)
    output_lines.append(f"TỔNG KẾT CHIẾN LƯỢC [{strategy_name.upper()}]:")
    output_lines.append(f"  - Điểm Retrieval (theo doc_id): {total_score}/{len(benchmark_queries) * 2} điểm")
    output_lines.append(f"  - Số câu chứa chính xác đáp án thực tế trong top-3: {total_content_hits}/{len(benchmark_queries)}")
    output_lines.append("=" * 65)

    full_output = "\n".join(output_lines)
    return full_output


def main():
    parser = argparse.ArgumentParser(description="Chạy RAG Benchmark trên bộ tài liệu học bổng.")
    parser.add_argument(
        "strategy",
        nargs="?",
        default="heading",
        choices=["heading", "recursive", "fixed", "sentences"],
        help="Chiến lược chunking cần đánh giá (default: heading)",
    )
    parser.add_argument(
        "--queries-file",
        "-q",
        default="data/scholarship/benchmark_gold_set.json",
        help="Đường dẫn file JSON chứa benchmark queries & gold answers",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        default="data/scholarship",
        help="Thư mục chứa các tài liệu .md",
    )

    args = parser.parse_args()

    result = run_benchmark(
        strategy_name=args.strategy,
        data_dir=args.data_dir,
        queries_file=args.queries_file,
    )
    print(result)

    output_file = Path("ket_qua_benchmark.txt")
    output_file.write_text(result, encoding="utf-8")
    print(f"\n[✓] Đã lưu kết quả chi tiết vào file: {output_file.resolve()}")


if __name__ == "__main__":
    main()

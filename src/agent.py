from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if self.store.get_collection_size() == 0:
            return "Không tìm thấy thông tin trong cơ sở tri thức (kho dữ liệu trống)."

        if not question or not question.strip():
            return "Vui lòng cung cấp câu hỏi."

        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức."

        context_blocks = []
        for idx, r in enumerate(results, start=1):
            meta = r.get("metadata", {})
            source = (
                meta.get("source")
                or meta.get("source_url")
                or meta.get("doc_id")
                or r.get("id", f"doc_{idx}")
            )
            content = r.get("content", "").strip()
            context_blocks.append(f"[{idx}] (Nguồn: {source})\n{content}")

        context_str = "\n\n".join(context_blocks)
        prompt = (
            "Bạn là một trợ lý hỏi đáp chuyên nghiệp. Hãy trả lời câu hỏi của người dùng "
            "DỰA HOÀN TOÀN vào các đoạn ngữ cảnh được cung cấp dưới đây.\n\n"
            f"--- NGỮ CẢNH ---\n{context_str}\n----------------\n\n"
            "Quy tắc bắt buộc:\n"
            "1. Chỉ sử dụng thông tin có trong ngữ cảnh. Không suy diễn hoặc tự bịa thông tin.\n"
            "2. Nếu ngữ cảnh không chứa thông tin để trả lời, hãy thông báo rõ ràng là không tìm thấy thông tin.\n"
            "3. Trích dẫn nguồn bằng số thứ tự [1], [2]... tương ứng với đoạn thông tin được dùng.\n\n"
            f"Câu hỏi: {question.strip()}\n"
            "Câu trả lời:"
        )

        return self.llm_fn(prompt)

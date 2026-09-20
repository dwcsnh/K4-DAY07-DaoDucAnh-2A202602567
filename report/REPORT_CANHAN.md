# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Đào Đức Anh (MSSV: 2A202602567)  
**Nhóm:** G16  
**Ngày:** 20/09/2026  

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiến gần về 1.0) nghĩa là hai vector embedding hướng về cùng một chiều trong không gian ngữ nghĩa đa chiều. Về mặt thông tin, điều này chứng minh hai đoạn văn bản có sự tương đồng sâu sắc về nội dung, ý nghĩa hoặc ngữ cảnh, ngay cả khi chúng sử dụng các từ vựng bề mặt hoàn toàn khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: *"Sinh viên có kết quả học tập và rèn luyện xuất sắc sẽ được nhà trường trao tặng học bổng khuyến khích."*
- Câu B: *"Người học đạt thành tích xuất chúng trong học kỳ được cấp kinh phí hỗ trợ tài chính để tuyên dương."*
- Tại sao tương đồng: Hai câu sử dụng tập từ vựng khác biệt ("sinh viên" vs "người học", "học tập xuất sắc" vs "thành tích xuất chúng", "trao tặng học bổng" vs "cấp kinh phí hỗ trợ tài chính"), nhưng truyền tải cùng một thông điệp chính sách khen thưởng. Mô hình embedding nắm bắt được mối quan hệ ngữ nghĩa tiềm ẩn (latent semantic relation) này nên hai vector có góc lệch cực nhỏ ($\cos \theta \approx 1$).

**Ví dụ có độ tương tự THẤP:**
- Câu A: *"Trường Đại học Công nghệ công bố điều kiện xét cấp học bổng khuyến khích học tập học kỳ 2."*
- Câu B: *"Dự báo thời tiết áp thấp nhiệt đới trên Biển Đông đang di chuyển nhanh theo hướng Tây Tây Bắc."*
- Tại sao khác: Hai câu thuộc hai miền kiến thức hoàn toàn độc lập (một bên là chế độ giáo dục/học bổng sinh viên, một bên là khí tượng thủy văn/thiên tai), không chia sẻ ngữ cảnh hay trường từ vựng liên quan, dẫn đến hai vector gần như vuông góc trong không gian embedding ($\cos \theta \approx 0$).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid nhạy cảm với độ dài hình học (magnitude) của vector — vốn bị chi phối bởi độ dài văn bản hoặc tần số từ lặp lại, khiến một câu ngắn và một đoạn văn dài cùng nghĩa có thể bị tính là ở xa nhau. Ngược lại, Cosine similarity chỉ đo góc giữa hai vector (phản ánh hướng ngữ nghĩa) mà không phụ thuộc vào độ dài. Đặc biệt khi vector đã được chuẩn hóa ($||v|| = 1$), Cosine similarity tương đương với tích vô hướng (dot product), cho phép tính toán cực nhanh trên quy mô lớn.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> Áp dụng công thức tính số chunk:
> $$N = \lceil \frac{\text{độ\_dài} - \text{overlap}}{\text{chunk\_size} - \text{overlap}} \rceil = \lceil \frac{10000 - 50}{500 - 50} \rceil = \lceil \frac{9950}{450} \rceil = \lceil 22.111... \rceil = 23$$
> Đã kiểm chứng thực tế qua `FixedSizeChunker(chunk_size=500, overlap=50).chunk('a'*10000)` cho kết quả chính xác 23 chunks.
> *Đáp án:* **23 chunks**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> - *Thay đổi số lượng chunk:* Áp dụng công thức với `overlap=100`:
>   $$N = \lceil \frac{10000 - 100}{500 - 100} \rceil = \lceil \frac{9900}{400} \rceil = \lceil 24.75 \rceil = 25 \text{ chunks}$$
>   Số lượng chunk tăng từ **23 lên 25 chunks** (tăng thêm 2 chunks).
> - *Lý do muốn độ chồng chéo nhiều hơn:*
>   1. **Tránh mất mát thông tin tại ranh giới cắt:** Giảm thiểu rủi ro một câu văn, điều khoản quan trọng hoặc bảng số liệu bị xé đôi giữa 2 chunk.
>   2. **Duy trì tính liên tục ngữ cảnh (Context Continuity):** Giúp chunk phía sau tiếp nối được tiền đề ý nghĩa của chunk phía trước, đảm bảo câu hỏi của người dùng khi truy xuất có thể tìm thấy trọn vẹn ngữ cảnh mà không bị cụt ý.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng biểu thức chính quy với kỹ thuật positive lookbehind `r'(?<=[.!?])\s+'` để nhận diện điểm ngắt câu sau các dấu `. `, `! `, `? `, `.\n` mà không làm mất dấu câu ở cuối câu trước. Xử lý các ngoại lệ như văn bản rỗng, chuỗi khoảng trắng trả về `[]`, và nhóm chính xác tối đa `max_sentences_per_chunk` câu vào mỗi chunk.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán kết hợp hai chiều: đệ quy xuống sâu theo thứ tự ưu tiên separator `["\n\n", "\n", ". ", " ", ""]` khi đoạn văn còn dài hơn `chunk_size`, và gom lũy tiến các mảnh nhỏ liền kề cho tới sát `chunk_size` để tránh chunk vụn 5–10 ký tự. Có 3 base cases: chuỗi rỗng (`[]`), chuỗi $\le$ `chunk_size` (`[text]`), và hết separator thì cắt cứng theo `chunk_size` để chống lỗi khi `separators=[]`.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Dữ liệu được lưu trữ in-memory trong danh sách `self._store` gồm các bản ghi chuẩn hóa sinh từ `_make_record(doc)`. Khi tìm kiếm, query được nhúng thành vector và tính điểm tương đồng với từng record qua tích vô hướng dot product (`_dot`), sau đó sắp xếp giảm dần theo điểm số để lấy top-k.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Thực hiện **tiền lọc (pre-filtering)** các ứng viên thỏa mãn toàn bộ cặp khóa-giá trị trong `metadata_filter` trước khi thực hiện similarity search để không làm mất slot top-k bởi các tài liệu sai điều kiện. Hàm `delete_document` lọc bỏ mọi chunk có `metadata['doc_id'] == doc_id` và trả về `True` nếu có bản ghi bị xóa, `False` nếu không tìm thấy.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Quy trình 3 bước: Kiểm tra store/câu hỏi rỗng $\rightarrow$ Truy xuất top-k chunk liên quan $\rightarrow$ Dựng prompt có cấu trúc trích dẫn `[1] (Nguồn: source)` và ràng buộc chống bịa (*anti-hallucination*), yêu cầu mô hình chỉ trả lời dựa trên ngữ cảnh và trích dẫn số thứ tự nguồn.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.08s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Học bổng khuyến khích học tập kỳ 2" | "Học bổng KKHT cho sinh viên cuối khóa" | cao | 0.82 | Đúng |
| 2 | "Điều kiện nhận học bổng Pegatron" | "Thời hạn nộp hồ sơ xin việc làm" | thấp | 0.24 | Đúng |
| 3 | "Thủ tục xét cấp học bổng Đinh Thiện Lý" | "Quy trình nộp bảng điểm và bài luận cá nhân" | cao | 0.76 | Đúng |
| 4 | "Chương trình đào tạo kỹ thuật tại Trung Quốc" | "Lịch thi học kỳ hè năm học 2026" | thấp | 0.15 | Đúng |
| 5 | "Mức tiền thưởng học bổng Vallet" | "Danh sách sinh viên nhận thưởng tại Văn Miếu" | cao | 0.71 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Cặp 1 và cặp 3 cho độ tương tự cao dù có nhiều từ viết tắt ("KKHT", "bài luận cá nhân"). Điều này cho thấy embedding có khả năng ánh xạ các khái niệm ngữ nghĩa tương đương vào cùng một vùng không gian vector đa chiều tốt hơn phương pháp so khớp từ khóa rời rạc (lexical search).

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi | Chunk được truy xuất hàng đầu (Top-1 Chunk trích yếu) | Điểm số (Cosine) | Có liên quan? (Yes/No) | Ghi chú / Đánh giá Agent |
|---|---------|-------------------------------------------------------|-------------------|------------------------|--------------------------|
| 1 | Mức học bổng khuyến khích học tập kì cuối cho sinh viên loại Xuất sắc ngành CLC CNTT khóa QH-2022 là bao nhiêu? | `cap-hoc-bong-khuyen-khich-hoc-tap-ki-cuoi-thang-06-2026`: Học bổng khuyến khích học tập được cấp theo định mức: Chương trình đào tạo Mức học bổng/sinh viên/tháng Loại Xuất sắc... 4.100.000... (kèm QH-2022) | 0.6994 | Có (Trúng Gold Doc & Chứa từ khóa đáp án) | Agent trả lời chính xác mức 4.100.000 đồng/SV/tháng theo QĐ 1545/QĐ-ĐHCN |
| 2 | Mỗi suất học bổng Vallet năm 2026 dành cho sinh viên Trường ĐHCN trị giá bao nhiêu và lễ trao học bổng diễn ra ở đâu? | `ket-qua-xet-chon-hoc-bong-vallet-2026`: # Kết quả xét chọn học bổng Vallet dành cho sinh viên Trường Đại học Công nghệ, Đại học Quốc gia Hà Nội năm 2026 Kết quả... | 0.6773 | Có (Trúng Gold Doc ở Top-1) | Trúng đúng tài liệu Vallet ở vị trí số 1; chi tiết trị giá 34 triệu và địa điểm Văn Miếu nằm ở các câu tiếp theo |
| 3 | Sinh viên cần đáp ứng những tiêu chuẩn gì về GPA, rèn luyện và ngoại ngữ để được xét chọn Học bổng Tài năng Pegatron 2027? | `chuong-trinh-hoc-bong-tai-nang-pegatron-2027`: # Chương trình Học bổng Tài năng Pegatron 2027... Tiêu chuẩn GPA từ 2,8 trở lên, kết quả rèn luyện loại Tốt, hoàn thành chứng chỉ tiếng Trung HSK3... | 0.7233 | Có (Trúng Gold Doc & Chứa từ khóa đáp án) | SentenceChunker gom được cả tiêu đề lẫn điều kiện, đưa tài liệu Pegatron lên thẳng Top-1 với score cao nhất (0.7233) |
| 4 | Hồ sơ đăng ký học bổng Đinh Thiện Lý dành cho sinh viên năm cuối gồm những giấy tờ gì và hạn nộp là khi nào? | `chuong-trinh-hoc-bong-dinh-thien-ly-nam-cuoi-2026-2027`: Hội đồng xét chọn... Đăng ký học bổng: bản scan pdf, bài luận tiếng Anh tối đa 500 từ... hạn trước 16h30 ngày 21/7/2026 | 0.6726 | Có (Trúng Gold Doc & Chứa từ khóa đáp án) | Chunk giữ trọn vẹn cả danh mục hồ sơ (500 từ) và thời hạn nộp 21/7/2026 |
| 5 | Chương trình học bổng Goertek năm 2027 gồm những mô hình đào tạo nào và quyền lợi của từng mô hình là gì? | `chuong-trinh-hoc-bong-goertek-nam-2027`: # Chương trình học bổng Goertek năm 2027... Mô hình đào tạo tại VN học bổng 60 triệu đồng... Đào tạo tại TQ học bổng 80 triệu đồng, vé máy bay khứ hồi... | 0.7160 | Có (Trúng Gold Doc & Chứa từ khóa đáp án) | Cả 2 mô hình đào tạo và mức hỗ trợ tài chính (60tr, 80tr) đều nằm trọn trong chunk Top-1 |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5 câu hỏi (Đạt **10/10 điểm Retrieval** tuyệt đối theo chuẩn `docs/SCORING.md`, trong đó **4/5** câu hỏi chứa trực tiếp từ khóa đáp án thực tế ngay tại Top-1).

**Điều hay nhất tôi học được từ việc chuyển đổi sang chiến lược SentenceChunker:**
> 1. **Bảo toàn ngữ cảnh câu hành chính:** Việc gom 3 câu (`max_sentences_per_chunk=3`) theo các ranh giới kết thúc tự nhiên (`. `, `! `, `? `, `.\n`) giúp các mệnh đề điều kiện học bổng (tiêu chuẩn GPA, rèn luyện, ngoại ngữ) không bị cắt rời khỏi tiêu đề tài liệu hay đơn vị đo lường, giải quyết triệt để ca trượt ở Câu 3 từng gặp phải khi phân chia theo độ dài ký tự thô.
> 2. **Tối ưu dung lượng và giảm nhiễu Vector Store:** Toàn bộ 10 tài liệu được gom thành đúng 40 chunk vừa vặn (thay vì hơn 100 chunk vụn), giảm mật độ nhiễu từ các đoạn footer website ("Bài viết liên quan"), giúp mô hình embedding `text-embedding-3-small` định vị chính xác tuyệt đối 100% tài liệu chuẩn ở vị trí Top-1.
> 3. **Hiệu quả của Metadata Pre-filtering:** Ở Câu 1, bộ lọc `{'audience': 'student'}` tiếp tục phát huy tác dụng loại bỏ các tài liệu quy chế chung, hỗ trợ SentenceChunker đạt độ chính xác tối đa.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |

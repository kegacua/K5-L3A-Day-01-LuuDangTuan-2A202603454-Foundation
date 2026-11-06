# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202603454**
- **Họ và tên: Lưu Đăng Tuấn**
- **SEED: 4110**

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | có |

## 2. Kết quả chính


Phương án	Precision	Recall	F1 (test)	F1 (SUB 40)	train (s)	tin/s (cả lô)	p50 ms/tin	p95 ms/tin	USD/1k tin
1. BoW + LogReg (tự code)	0.986	0.818	0.894	0.974	5.543	8078.394	0.038	0.084	0.0
2. BoW + MLP (deep learning)	0.963	0.886	0.923	1.000	9.229	33467.378	0.232	0.869	0.0
3a. Encoder (MiniLM) + kNN	0.854	0.932	0.891	0.974	70.630	40.563	30.801	46.072	0.0
3b. Encoder (MiniLM) + LogReg	0.942	0.920	0.931	0.974	70.907	43.125	28.217	39.921	0.0
3c. Encoder (MiniLM) + MLP head	0.976	0.920	0.947	0.974	71.445	43.968	25.659	38.367	0.0

## 3. Dự đoán và thực tế

1.ml_f1 — Dự đoán: "F1 test ~0.85–0.95, precision cao hơn recall vì dữ liệu lệch."

Thực tế: F1 = 0.894 (đúng khoảng), precision 0.986 > recall 0.818 (đúng chiều).

Lệch nhẹ: Khoảng dự đoán đúng nhưng tôi ngầm kỳ vọng F1 gần 0.92–0.93 hơn; thực tế mô hình bỏ sót khá nhiều spam (recall 0.818).

2.dl_curve — Dự đoán: "Train loss giảm liên tục về 0 do MLP hơn 128.000 tham số."

Thực tế: Train loss giảm, nhưng val loss chạm đáy ở epoch 10 rồi tăng → overfit từ epoch ~11. F1 test (0.923) chỉ nhỉnh hơn LogReg (0.894) chút.

Lệch: Tôi chỉ dự đoán phần train loss mà bỏ qua val loss; bài học: với BoW thưa và dữ liệu nhỏ, MLP không "vượt trội" như kỳ vọng.

3.emb_tricky — Dự đoán: "Encoder + LogReg đúng nhiều hơn BoW."

Thực tế: Cả hai đều đúng 5/6 case; case "Last chance! Your account rewards expire tonight, confirm details" bị cả hai đoán sai (ham).

Lệch: Embedding không vượt trội trên bộ 6 case nhỏ; khoảng cách 0 là do mẫu quá ít, không đủ kết luận.

## 4. Phân tích 50 case khó

Model tụt nhiều nhất: BoW + LogReg / BoW + MLP — F1 từ ~0.97–1.00 rơi xuống 0.38–0.41 (mất ~55–60 điểm). Lý do: BoW chỉ nhìn token rời rạc, nên (a) các tin "tưởng spam nhưng không phải" chứa từ khoá mạnh bị gán spam, (b) spam "né từ khoá" bị bỏ qua vì token lạ không có trong vocab.

Hai case cụ thể khiến model sai:

Case #12 — "hi there, sorry to bother you, my friend gave me your number, I'm new in town and looking for someone to chat with, check my profile at meetlocal.club" — nhãn thật = spam, tất cả model đều sai. Cơ chế: không có từ khoá spam nào; BoW không có tín hiệu; embedding/kNN gần với các tin ham "kết bạn" nên đoán ham. Đây là spam "lừa tình/dating" thuần ngữ nghĩa, cần suy luận về ý đồ (link lạ + mời chat) mà cả 3 thế hệ đều thiếu.

Case #33 — "Get your l0an appr0ved in 5 m1nutes, n0 credit check. Repl¥ L0AN" — nhãn thật = spam, 3c sai. Cơ chế: l0an, appr0ved, Repl¥ là token OOV với BoW; embedding MiniLM cũng bị nhiễu vì ký tự thay thế phá vỡ subword quen thuộc → vector lệch khỏi cụm spam. Đây là lý do cần bước chuẩn hoá văn bản (de-obfuscation) trước khi đưa vào model.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

Embedding (MiniLM) + LogReg — hoặc BoW + MLP nếu muốn tối ưu chi phí hạ tầng.

Dẫn số liệu:

3b đạt p95 = 39.9 ms (đạt SLA), F1 test 0.931 — chất lượng cao nhất trong nhóm đạt SLA.

3c đạt F1 0.947 nhưng p95 = 38.4 ms cũng đạt SLA → nếu ưu tiên chất lượng tuyệt đối thì chọn 3c, đánh đổi thêm ~0.4s train và độ phức tạp.

2 (BoW + MLP) cực nhanh (p95 0.87 ms, throughput 33k tin/s) và miễn phí, F1 0.923 — nhưng cần vectorize 2000 chiều, khó mở rộng ngữ nghĩa.

LLM: loại — p95 hàng chục–hàng trăm ms, vượt SLA; chi phí ước tính cho 10M tin/ngày với gpt-6-luna (~$0.10/1M input, 100 token input + 1 output/tin): 10M × 100 / 1M × $0.10 = $1/ngày input** + output không đáng kể ≈ **$30/tháng, nhưng độ trễ không đạt SLA nên không dùng làm lớp chính.

Đánh đổi chấp nhận: 3b chậm hơn BoW ~10.000 lần mỗi tin, cần GPU/hạ tầng encode; nhưng F1 cao hơn 3.7 điểm và tổng quát hoá tốt hơn với spam mới (nhờ không phụ thuộc vocab cố định). Chấp nhận encode theo lô (batch) để tăng throughput.

## 7. AI log và điều đã học
Phần	Hỏi AI gì	AI trả lời/viết gì	Đã sửa/kiểm chứng
Phần 4	Viết call_openai với provider_kwargs cho GPT-6	Gợi ý max_completion_tokens + reasoning_effort="none"	Test mock 4.1 pass
Phần 4	parse_label xử lý "Not spam"	Tách token, xét "not"+"spam" liền kề	Tự viết lại, test pass
Phần 5	Đọc biểu đồ Pareto	Giải thích dominated/dominating	Đối chiếu output thực tế

Điều đã học:

Cách biểu diễn văn bản quan trọng hơn độ sâu mạng — MLP trên BoW chỉ hơn LogReg ~3 điểm F1, trong khi embedding + LogReg đạt 0.931.

Điểm test cùng phân phối quá lạc quan — F1 0.97 rơi còn 0.38–0.68 trên 50 case khó; cần hard eval trước khi tin một con số F1.

LLM không phải vạn năng — chậm, đắt, dễ bị prompt injection; hợp lý nhất là fallback cho vùng lưỡng lự, không thay lớp ML chính khi có SLA nghiêm ngặt.


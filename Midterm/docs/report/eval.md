Viewed benchmark_results_partial_500sam.csv:1967-2002

Dạ đúng rồi bạn, **EM (Exact Match) là chỉ số nhị phân (chỉ có 0 hoặc 1)**:
* Nếu câu dự đoán giống **100%** đáp án gốc (sau khi đã chuẩn hóa chữ thường và bỏ dấu câu) $\rightarrow$ Điểm = **1**.
* Nếu sai lệch dù chỉ 1 từ, 1 chữ $\rightarrow$ Điểm = **0**.

Hãy nhìn ngay vào dữ liệu thực tế từ file kết quả của bạn để thấy rõ sự khác biệt giữa **EM** và **F1**:

#### Ví dụ 1 (Dòng 1981):
* **Câu hỏi:** *Which of the following pathologies is present in this image?* (Trắc nghiệm)
* **Đáp án gốc (Gold):** `A`
* **Model dự đoán (Pred):** `A) nodule`
* **Điểm số:**
  * **EM = 0** (Vì `"a"` không khớp hoàn toàn với `"a nodule"` $\rightarrow$ Sai lệch hoàn toàn).
  * **F1 = 0.6667** (Vì mô hình đoán đúng được ký tự `A` nên vẫn được vớt điểm khớp một phần).

#### Ví dụ 2 (Dòng 1987):
* **Câu hỏi:** *Does the patient have urolithiasis?* (Câu hỏi Yes/No)
* **Đáp án gốc (Gold):** `no`
* **Model dự đoán (Pred):** `no`
* **Điểm số:**
  * **EM = 1** (Khớp hoàn toàn).
  * **F1 = 1.0** (Khớp hoàn toàn).

---

### Tóm lại:
* **EM = 1** đòi hỏi sự hoàn hảo tuyệt đối (chấm điểm gắt).
* **F1** thì linh hoạt hơn, cho phép mô hình viết thừa từ hoặc thiếu từ mà vẫn có điểm (chấm điểm nương tay, phản ánh đúng ngữ nghĩa hơn cho các câu hỏi mở).



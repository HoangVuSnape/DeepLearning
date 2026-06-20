# 📊 Slide Material: Model Configurations & Dataset Details

Tài liệu tóm tắt cấu hình mô hình (Gemma-4 E2B/E4B) và thông tin bộ dữ liệu phục vụ cho việc làm slide báo cáo.

---

## 1. ⚙️ Cấu Hình Chi Tiết của 2 Model (LoRA & Training Hyperparameters)

Dưới đây là bảng thông số cấu hình tham số được sử dụng để tinh chỉnh hai mô hình:

### A. Cấu hình LoRA (LoRA Configuration)
| Cấu hình LoRA | Gemma-4 E2B (2B) | Gemma-4 E4B (4B) |
| :--- | :---: | :---: |
| **LoRA Rank (r)** | `16` | `16` |
| **LoRA Alpha** | `16` | `16` |
| **LoRA Dropout** | `0.0` (0) | `0.0` (0) |
| **LoRA Bias** | `"none"` | `"none"` |
| **Target Modules** | Cả phần ngôn ngữ và hình ảnh (Attention & MLP: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) | Cả phần ngôn ngữ và hình ảnh (Attention & MLP: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) |

### B. Siêu tham số huấn luyện (Training Hyperparameters)
| Siêu tham số | Gemma-4 E2B (2B) | Gemma-4 E4B (4B) |
| :--- | :---: | :---: |
| **Learning Rate (Tốc độ học)** | `2e-4` (`0.0002`) | `2e-4` (`0.0002`) |
| **Optimizer (Thuật toán tối ưu)** | `AdamW (8-bit)` (`adamw_8bit`) | `AdamW (8-bit)` (`adamw_8bit`) |
| **Batch Size per Device** | `1` | `1` |
| **Gradient Accumulation Steps** | `4` | `4` |
| **Effective Batch Size (Tổng)** | `4` | `4` |
| **LR Scheduler** | `Cosine` (`cosine`) | `Cosine` (`cosine`) |
| **Số lượng Epochs** | `2` | `2` |
| **Tổng số Steps (Max Steps)** | `898` | `898` |
| **Logging Steps** | `10` | `10` |
| **Lưu Checkpoint (Save Steps)** | `50` | `50` |
| **Tối ưu hóa VRAM (Unsloth)** | Tải mô hình dạng `4-bit` (`load_in_4bit=True`) | Tải mô hình dạng `4-bit` (`load_in_4bit=True`) |

---

## 2. 📂 Thông Tin Bộ Dữ Liệu (Dataset Details)

Mô hình được huấn luyện và đánh giá trên bộ dữ liệu **VQA-RAD** (Visual Question Answering in Radiology) - tập dữ liệu chuẩn mực nhất trong hỏi đáp hình ảnh y khoa lâm sàng.

### 📊 Thống kê Số lượng (Data Splits):
* **Tổng số mẫu:** 2,244 cặp câu hỏi - trả lời sau khi lọc trùng.
* **Tập Huấn luyện (Train Split):** 1,793 mẫu (chiếm ~80%).
* **Tập Kiểm thử (Test Split):** 451 mẫu (chiếm ~20%).
* **Số lượng hình ảnh lâm sàng gốc:** 314 ảnh (chụp X-quang, CT, MRI từ cơ sở dữ liệu MedPix).

### 📋 Cấu trúc Dữ liệu (Features):
Mỗi mẫu dữ liệu chứa 3 thuộc tính chính:
1. `image`: Ảnh y khoa gốc chụp thực tế (X-ray, CT, MRI).
2. `question`: Câu hỏi lâm sàng bằng tiếng Anh (ví dụ: *"Is there cardiomegaly?"*, *"What organ is shown?"*).
3. `answer`: Câu trả lời ngắn trực tiếp do bác sĩ lâm sàng gán nhãn (ví dụ: *"yes"*, *"no"*, *"chest x-ray"*).

### 🏷️ Phân loại câu hỏi trong đánh giá:
Do bộ dữ liệu gốc không có nhãn loại câu hỏi sẵn, trong quá trình chạy kiểm thử, câu hỏi được tự động phân thành 2 nhóm:
* **Closed-ended (Yes/No):** Chiếm **251 mẫu** (khoảng ~56% tập test). Trả lời trực tiếp dạng nhị phân.
* **Open-ended (Câu hỏi tự do):** Chiếm **200 mẫu** (khoảng ~44% tập test). Trả lời tên bộ phận cơ thể hoặc loại bệnh lý.

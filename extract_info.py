import json
import re
import os

notebook_paths = {
    "Gemma-4 E2B (2B) Fine-Tuning": "Midterm/vqa_final/v3-vqa-2b-it.ipynb",
    "Gemma-4 E4B (4B) Fine-Tuning": "Midterm/vqa_final/v3-vqa-4b-it.ipynb"
}

output_md_path = "Midterm/docs/report/slide_material.md"

def extract_configs_from_notebook(nb_path):
    if not os.path.exists(nb_path):
        return f"⚠️ File không tồn tại: {nb_path}"
        
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    code_text = ""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            code_text += "\n" + "".join(cell.get("source", []))
            
    # Regex patterns to find parameters
    patterns = {
        "LoRA r (Rank)": r"\br\s*=\s*(\d+)",
        "LoRA Alpha": r"\blora_alpha\s*=\s*(\d+)",
        "LoRA Target Modules": r"\btarget_modules\s*=\s*(\[[^\]]+\])",
        "LoRA Dropout": r"\blora_dropout\s*=\s*([\d\.]+)",
        "LoRA Bias": r"\bbias\s*=\s*['\"]([^'\"]+)['\"]",
        "Gradient Checkpointing": r"\buse_gradient_checkpointing\s*=\s*['\"]([^'\"]+)['\"]|\buse_gradient_checkpointing\s*=\s*(True|False)",
        "Learning Rate": r"\blearning_rate\s*=\s*([\d\.\-e]+)",
        "Batch Size per Device": r"\bper_device_train_batch_size\s*=\s*(\d+)",
        "Gradient Accumulation Steps": r"\bgradient_accumulation_steps\s*=\s*(\d+)",
        "Optimizer": r"\boptim\s*=\s*['\"]([^'\"]+)['\"]",
        "Weight Decay": r"\bweight_decay\s*=\s*([\d\.]+)",
        "LR Scheduler": r"\blr_scheduler_type\s*=\s*['\"]([^'\"]+)['\"]",
        "Logging Steps": r"\blogging_steps\s*=\s*(\d+)",
        "Epochs": r"\bnum_train_epochs\s*=\s*(\d+)",
        "Max Steps": r"\bmax_steps\s*=\s*(\d+)",
        "Warmup Steps": r"\bwarmup_steps\s*=\s*(\d+)",
        "Warmup Ratio": r"\bwarmup_ratio\s*=\s*([\d\.]+)",
        "Seed": r"\bseed\s*=\s*(\d+)",
    }
    
    extracted = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, code_text)
        if match:
            # Get first non-empty group
            val = next((g for g in match.groups() if g is not None), "N/A")
            extracted[name] = val
        else:
            extracted[name] = "Mặc định / Không tìm thấy"
            
    return extracted

# Extracting configs
results = {}
for name, path in notebook_paths.items():
    results[name] = extract_configs_from_notebook(path)

# Writing Markdown Slide Material
md_content = """# 📊 Slide Material: Model Configurations & Dataset Details

Tài liệu tóm tắt cấu hình mô hình (Gemma-4 2B/4B) và thông tin bộ dữ liệu để chuẩn bị nội dung làm slide báo cáo.

---

## 1. ⚙️ Cấu Hình Chi Tiết của 2 Model (LoRA & Training Hyperparameters)

Dưới đây là bảng so sánh cấu hình tham số được sử dụng để tinh chỉnh hai mô hình:

| Tham số / Cấu hình | Gemma-4 E2B (2B) | Gemma-4 E4B (4B) |
| :--- | :--- | :--- |
"""

# Dynamic generation of comparative table rows
all_keys = [
    "LoRA r (Rank)", "LoRA Alpha", "LoRA Target Modules", "LoRA Dropout", "LoRA Bias", "Gradient Checkpointing",
    "Learning Rate", "Batch Size per Device", "Gradient Accumulation Steps", "Optimizer", "Weight Decay",
    "LR Scheduler", "Epochs", "Max Steps", "Warmup Steps", "Warmup Ratio", "Seed"
]

for key in all_keys:
    val_2b = results["Gemma-4 E2B (2B) Fine-Tuning"].get(key, "N/A") if isinstance(results["Gemma-4 E2B (2B) Fine-Tuning"], dict) else "Error"
    val_4b = results["Gemma-4 E4B (4B) Fine-Tuning"].get(key, "N/A") if isinstance(results["Gemma-4 E4B (4B) Fine-Tuning"], dict) else "Error"
    md_content += f"| **{key}** | `{val_2b}` | `{val_4b}` |\n"

md_content += """
> 📌 *Ghi chú:* 
> * Cả hai mô hình đều được tối ưu hóa bằng **Unsloth** để chạy trên VRAM thấp (tự động tải dạng 4-bit qua `load_in_4bit=True`).
> * Các `target_modules` bao gồm các lớp attention chính (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) để tối ưu hóa hiệu quả tinh chỉnh LoRA.

---

## 2. 📂 Thông Tin Bộ Dữ Liệu (Dataset Details)

Hệ thống được huấn luyện và đánh giá trên bộ dữ liệu **VQA-RAD** (Visual Question Answering in Radiology) - tập dữ liệu chuẩn trong lĩnh vực hỏi đáp hình ảnh y khoa.

### 📊 Thống kê Số lượng (Data Splits):
* **Tổng số mẫu:** 2,244 cặp câu hỏi - trả lời.
* **Tập Huấn luyện (Train Split):** 1,793 mẫu (chiếm ~80%).
* **Tập Kiểm thử (Test Split):** 451 mẫu (chiếm ~20%).
* **Số lượng hình ảnh gốc:** 315 ảnh chụp X-quang, CT, MRI thực tế từ các ca lâm sàng.

### 📋 Cấu trúc Dữ liệu (Features):
Mỗi mẫu dữ liệu chứa 3 thuộc tính chính:
1. `image`: Ảnh y khoa chụp thực tế (X-ray, CT, MRI).
2. `question`: Câu hỏi bằng tiếng Anh (ví dụ: *"Is there cardiomegaly?"*, *"What organ is shown?"*).
3. `answer`: Câu trả lời ngắn trực tiếp (ví dụ: *"yes"*, *"no"*, *"chest x-ray"*).

### 🏷️ Phân loại câu hỏi trong đánh giá:
Do bộ dữ liệu gốc không có nhãn phân loại sẵn, trong quá trình kiểm thử, câu hỏi được tự động phân thành 2 nhóm:
* **Closed-ended (Yes/No):** Chiếm khoảng 60-70% tập dữ liệu. Trả lời trực tiếp dạng nhị phân.
* **Open-ended (Câu hỏi tự do):** Chiếm khoảng 30-40% tập dữ liệu. Trả lời tên bộ phận cơ thể, loại bệnh lý hoặc tư thế chụp.
"""

os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
with open(output_md_path, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"✅ Đã tạo file slide material thành công tại: {output_md_path}")
print("Hãy chạy lệnh: python extract_info.py để cập nhật dữ liệu từ notebook.")

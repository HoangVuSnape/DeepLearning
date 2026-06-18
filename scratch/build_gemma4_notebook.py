import json
import os

notebook_path = r"e:\DoCode\Master_Subject2026_TDTU\DeepLearning\Midterm\Notebook\Gemma4_VQA_Radiology.ipynb"

# Ensure directory exists
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

cells = []

# Cell 1: Intro (Markdown)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Gemma-4 Medical Vision-Language VQA Fine-Tuning with Unsloth\n",
        "\n",
        "Notebook này hướng dẫn tinh chỉnh (fine-tune) mô hình ngôn ngữ lớn đa phương tiện **Gemma-4 E2B/E4B** trên tập dữ liệu **Radiology Mini** để thực hiện tác vụ Medical Visual Question Answering (VQA).\n",
        "Hệ thống tích hợp giám sát tự động bằng **Comet ML** và thông báo tiến trình huấn luyện thời gian thực thông qua **Discord Webhook**.\n",
        "\n",
        "*   **Mô hình mặc định:** `unsloth/gemma-4-E2B-it` (có cấu hình sẵn cho `unsloth/gemma-4-E4B-it` ở dạng comment/ẩn đi).\n",
        "*   **Tập dữ liệu:** `unsloth/Radiology_mini` (X-ray images & captioning).\n",
        "*   **Công nghệ tối ưu:** Unsloth & SFTTrainer giúp giảm VRAM cực thấp."
    ]
})

# Cell 2: Installation (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%%capture\n",
        "import os, re\n",
        "# Cài đặt thư viện Unsloth cùng các dependency cần thiết\n",
        "if \"COLAB_\" not in \"\".join(os.environ.keys()):\n",
        "    !pip install unsloth  # Chạy local hoặc các dịch vụ đám mây khác\n",
        "else:\n",
        "    import torch; v = re.match(r'[\\d]{1,}\\.[\\d]{1,}', str(torch.__version__)).group(0)\n",
        "    xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, \"0.0.34\")\n",
        "    !pip install sentencepiece protobuf \"datasets==4.3.0\" \"huggingface_hub>=0.34.0\" hf_transfer\n",
        "    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth\n",
        "    !pip install --no-deps --upgrade \"torchao>=0.16.0\"\n",
        "!pip install --no-deps transformers==5.5.0 \"tokenizers>=0.22.0,<=0.23.0\"\n",
        "!pip install torchcodec\n",
        "!pip install --no-deps --upgrade timm\n",
        "!pip install comet-ml requests pillow"
    ]
})

# Cell 3: Secrets & Comet ML (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "import comet_ml\n",
        "\n",
        "# 1. Tải Comet ML API Key & Discord Webhook từ Kaggle Secrets nếu có\n",
        "try:\n",
        "    from kaggle_secrets import UserSecretsClient\n",
        "    user_secrets = UserSecretsClient()\n",
        "    os.environ[\"COMET_API_KEY\"] = user_secrets.get_secret(\"COMET_API_KEY\")\n",
        "    DISCORD_WEBHOOK_URL = user_secrets.get_secret(\"DISCORD_WEBHOOK_URL\")\n",
        "    print(\"🔑 Đã nạp thành công API Key & Webhook từ Kaggle Secrets!\")\n",
        "except Exception as e:\n",
        "    # Nếu chạy local hoặc không dùng Kaggle Secrets\n",
        "    os.environ[\"COMET_API_KEY\"] = \"YOUR_COMET_API_KEY\"  # Điền API Key của Comet ML tại đây\n",
        "    DISCORD_WEBHOOK_URL = \"YOUR_DISCORD_WEBHOOK_URL\"    # Điền Webhook Discord tại đây\n",
        "    print(\"⚠️ Không dùng Kaggle Secrets. Vui lòng kiểm tra điền tay trong code.\")\n",
        "\n",
        "# 2. Cấu hình dự án Comet ML\n",
        "os.environ[\"COMET_PROJECT_NAME\"] = \"gemma4-medical-vqa\"\n",
        "comet_ml.init(project_name=\"gemma4-medical-vqa\")"
    ]
})

# Cell 4: Data Prep Title (Markdown)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 1. Tải và Khám phá Dữ liệu (Data Prep & Processing)\n",
        "\n",
        "Chúng ta tải dữ liệu `unsloth/Radiology_mini` từ Hugging Face. Tập dữ liệu này chứa các hình ảnh X-ray lồng ngực kèm các caption mô tả chuyên sâu về y học của bác sĩ chẩn đoán hình ảnh."
    ]
})

# Cell 5: Load dataset (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from datasets import load_dataset\n",
        "\n",
        "# Tải dataset\n",
        "dataset = load_dataset(\"unsloth/Radiology_mini\", split=\"train\")\n",
        "print(\"📊 Thông tin dataset:\")\n",
        "print(dataset)\n",
        "print(f\"\\n👉 Số lượng mẫu dữ liệu: {len(dataset)}\")"
    ]
})

# Cell 6: Inspect a sample (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Trực quan mẫu dữ liệu thứ 3\n",
        "sample_idx = 2\n",
        "sample = dataset[sample_idx]\n",
        "print(f\"🔍 Mẫu index {sample_idx}:\")\n",
        "print(f\"• Image ID: {sample['image_id']}\")\n",
        "print(f\"• CUI (Medical Concepts): {sample['cui']}\")\n",
        "print(f\"• Mô tả bệnh án (Caption): {sample['caption']}\")\n",
        "\n",
        "# Hiển thị hình ảnh X-ray\n",
        "display(sample[\"image\"].resize((300, 300)))"
    ]
})

# Cell 7: Data Format Explain (Markdown)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "Mô hình ngôn ngữ hình ảnh (VLM) của Unsloth Gemma-4 yêu cầu dữ liệu chuẩn hóa dạng hội thoại (`messages`). Cấu trúc gồm ảnh và câu hỏi ở role `user`, câu trả lời ở role `assistant`:\n",
        "```python\n",
        "[\n",
        "    {\n",
        "        \"role\": \"user\",\n",
        "        \"content\": [\n",
        "            {\"type\": \"text\", \"text\": \"Describe what you see in this radiology image.\"},\n",
        "            {\"type\": \"image\", \"image\": sample[\"image\"]}\n",
        "        ]\n",
        "    },\n",
        "    {\n",
        "        \"role\": \"assistant\",\n",
        "        \"content\": [\n",
        "            {\"type\": \"text\", \"text\": sample[\"caption\"]}\n",
        "        ]\n",
        "    }\n",
        "]\n",
        "```"
    ]
})

# Cell 8: Data Conversion (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "instruction = \"Describe what you see in this radiology image.\"\n",
        "\n",
        "def convert_to_conversation(sample):\n",
        "    conversation = [\n",
        "        {\n",
        "            \"role\": \"user\",\n",
        "            \"content\": [\n",
        "                {\"type\": \"text\", \"text\": instruction},\n",
        "                {\"type\": \"image\", \"image\": sample[\"image\"]}\n",
        "            ]\n",
        "        },\n",
        "        {\n",
        "            \"role\": \"assistant\",\n",
        "            \"content\": [{\"type\": \"text\", \"text\": sample[\"caption\"]}]\n",
        "        }\n",
        "    ]\n",
        "    return {\"messages\": conversation}\n",
        "\n",
        "# Tiến hành ánh xạ toàn bộ tập dữ liệu\n",
        "converted_dataset = [convert_to_conversation(item) for item in dataset]\n",
        "print(f\"✅ Đã xử lý và chuyển đổi định dạng {len(converted_dataset)} mẫu dữ liệu hội thoại.\")\n",
        "print(\"👉 Xem thử mẫu đầu tiên:\")\n",
        "import pprint\n",
        "pprint.pprint(converted_dataset[0][\"messages\"][0])"
    ]
})

# Cell 9: Model Setup Title (Markdown)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Khởi tạo Mô hình Gemma-4 (Model & LoRA Setup)\n",
        "\n",
        "Chúng ta load model Gemma-4. Theo yêu cầu, mô hình **Gemma-4 E2B** được chạy mặc định, còn mô hình lớn hơn **Gemma-4 E4B** được ẩn đi dưới dạng comment để dễ dàng kích hoạt khi cần."
    ]
})

# Cell 10: Model loading (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from unsloth import FastVisionModel\n",
        "import torch\n",
        "\n",
        "# Chọn 1 trong 2 model. Model 4B được ẩn đi (comment out).\n",
        "model_name = \"unsloth/gemma-4-E2B-it\"  # Mô hình E2B (2 Tỷ tham số) - Nhẹ, debug nhanh\n",
        "# model_name = \"unsloth/gemma-4-E4B-it\"  # Mô hình E4B (4 Tỷ tham số) - Hiệu năng cao hơn\n",
        "\n",
        "print(f\"🚀 Đang tải mô hình từ HuggingFace: {model_name}...\")\n",
        "\n",
        "model, processor = FastVisionModel.from_pretrained(\n",
        "    model_name,\n",
        "    load_in_4bit = True,  # Sử dụng định dạng 4bit để siêu tiết kiệm VRAM\n",
        "    use_gradient_checkpointing = \"unsloth\",\n",
        ")"
    ]
})

# Cell 11: PEFT LoRA (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Cấu hình LoRA Adapter tiết kiệm tham số (chỉ train ~1.1% tổng trọng số)\n",
        "model = FastVisionModel.get_peft_model(\n",
        "    model,\n",
        "    finetune_vision_layers     = True,  # Fine-tune module hình ảnh (Vision encoder)\n",
        "    finetune_language_layers   = True,  # Fine-tune module ngôn ngữ (LLM)\n",
        "    finetune_attention_modules = True,  # Fine-tune các lớp attention\n",
        "    finetune_mlp_modules       = True,  # Fine-tune các lớp MLP\n",
        "\n",
        "    r = 16,                            # Độ rank của LoRA\n",
        "    lora_alpha = 16,                   # Alpha scale\n",
        "    lora_dropout = 0,\n",
        "    bias = \"none\",\n",
        "    random_state = 3407,\n",
        "    use_rslora = False,\n",
        "    target_modules = \"all-linear\",     # Áp dụng lên toàn bộ linear layers để chất lượng tối ưu\n",
        ")"
    ]
})

# Cell 12: Chat template apply (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Áp dụng template chat chuẩn của Gemma-4\n",
        "from unsloth import get_chat_template\n",
        "\n",
        "processor = get_chat_template(\n",
        "    processor,\n",
        "    chat_template = \"gemma-4\"\n",
        ")"
    ]
})

# Cell 13: Monitor Section (Markdown)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Cấu hình Callback Giám sát gửi thông báo về Discord\n",
        "\n",
        "Chúng ta định nghĩa lớp callback tùy chỉnh `DiscordWebhookCallback` kế thừa từ `TrainerCallback` của Hugging Face. Khi Trainer chạy qua các mốc bắt đầu, log loss định kỳ và kết thúc, tin nhắn sẽ tự động bắn về kênh Discord thông qua Webhook URL."
    ]
})

# Cell 14: Discord Callback Code (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import requests\n",
        "from transformers import TrainerCallback\n",
        "\n",
        "class DiscordWebhookCallback(TrainerCallback):\n",
        "    def __init__(self, webhook_url, model_name):\n",
        "        self.webhook_url = webhook_url\n",
        "        self.model_name = model_name\n",
        "\n",
        "    def send_discord_message(self, content):\n",
        "        if not self.webhook_url or \"discord.com/api/webhooks\" not in self.webhook_url:\n",
        "            print(\"⚠️ Discord Webhook chưa được thiết lập chính xác hoặc bị bỏ trống. Bỏ qua gửi thông báo!\")\n",
        "            return\n",
        "        try:\n",
        "            payload = {\"content\": content}\n",
        "            response = requests.post(self.webhook_url, json=payload)\n",
        "            if response.status_code != 204:\n",
        "                print(f\"⚠️ Gửi thông báo đến Discord thất bại: HTTP {response.status_code}\")\n",
        "        except Exception as e:\n",
        "            print(f\"❌ Lỗi kết nối khi gửi Discord: {e}\")\n",
        "\n",
        "    def on_train_begin(self, args, state, control, **kwargs):\n",
        "        self.send_discord_message(\n",
        "            f" + 'f"🚀 **[BẮT ĐẦU TRAINING]**\\n"' + "\n",
        "            f" + 'f"• **Mô hình:** `{self.model_name}`\\n"' + "\n",
        "            f" + 'f"• **Tham số:** `lr={args.learning_rate}`, `steps={args.max_steps}`, `batch_size={args.per_device_train_batch_size}`\\n"' + "\n",
        "            f" + 'f"• **Platform:** Comet ML Project: `gemma4-medical-vqa`"' + "\n",
        "        )\n",
        "\n",
        "    def on_log(self, args, state, control, logs=None, **kwargs):\n",
        "        # Định kỳ mỗi 10 steps sẽ gửi thông báo cập nhật một lần tránh bị Discord rate-limit\n",
        "        if logs and \"loss\" in logs:\n",
        "            step = state.global_step\n",
        "            loss = logs[\"loss\"]\n",
        "            epoch = logs.get(\"epoch\", 0.0)\n",
        "            lr = logs.get(\"learning_rate\", 0.0)\n",
        "            if step % 10 == 0 or step == args.max_steps:\n",
        "                self.send_discord_message(\n",
        "                    f" + 'f"📈 **[TIẾN TRÌNH TRAINING]** Step `{step}`/`{args.max_steps}` (Epoch {epoch:.2f})\\n"' + "\n",
        "                    f" + 'f"• **Loss:** `{loss:.4f}`\\n"' + "\n",
        "                    f" + 'f"• **Learning Rate:** `{lr:.2e}`"' + "\n",
        "                )\n",
        "\n",
        "    def on_train_end(self, args, state, control, **kwargs):\n",
        "        self.send_discord_message(\n",
        "            f" + 'f"🎉 **[TRAINING HOÀN TẤT]**\\n• Mô hình `{self.model_name}` đã train thành công với `{state.global_step}` steps!"' + "\n",
        "        )"
    ]
})

# Cell 15: Trainer setup explanation (Markdown)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. Huấn luyện Mô hình (Fine-Tuning Loop)\n",
        "\n",
        "Sử dụng `SFTTrainer` từ TRL cùng collator dữ liệu vision chuyên biệt `UnslothVisionDataCollator` của Unsloth để bắt đầu training."
    ]
})

# Cell 16: SFTTrainer config (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from unsloth.trainer import UnslothVisionDataCollator\n",
        "from trl import SFTTrainer, SFTConfig\n",
        "\n",
        "# Khởi tạo callback Discord gửi webhook\n",
        "discord_cb = DiscordWebhookCallback(webhook_url=DISCORD_WEBHOOK_URL, model_name=model_name)\n",
        "\n",
        "trainer = SFTTrainer(\n",
        "    model = model,\n",
        "    train_dataset = converted_dataset,\n",
        "    processing_class = processor.tokenizer,\n",
        "    data_collator = UnslothVisionDataCollator(model, processor),\n",
        "    callbacks = [discord_cb],\n",
        "    args = SFTConfig(\n",
        "        per_device_train_batch_size = 1,\n",
        "        gradient_accumulation_steps = 4,\n",
        "        max_grad_norm = 0.3,\n",
        "        warmup_ratio = 0.03,\n",
        "        max_steps = 60,         # Thiết lập 60 steps để demo nhanh, hãy đặt num_train_epochs=1 nếu train đầy đủ\n",
        "        learning_rate = 2e-4,\n",
        "        logging_steps = 1,\n",
        "        save_strategy = \"steps\",\n",
        "        optim = \"adamw_8bit\",\n",
        "        weight_decay = 0.001,\n",
        "        lr_scheduler_type = \"cosine\",\n",
        "        seed = 3407,\n",
        "        output_dir = \"outputs\",\n",
        "        report_to = \"comet\",    # Comet ML sẽ tự động bắt sự kiện và hiển thị biểu đồ loss!\n",
        "\n",
        "        # Các thiết lập bắt buộc phục vụ vision fine-tuning:\n",
        "        remove_unused_columns = False,\n",
        "        dataset_text_field = \"\",\n",
        "        dataset_kwargs = {\"skip_prepare_dataset\": True},\n",
        "        max_length = 2048,\n",
        "    )\n",
        ")"
    ]
})

# Cell 17: GPU stats display (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Kiểm tra bộ nhớ VRAM của GPU hiện hành trước khi chạy\n",
        "gpu_stats = torch.cuda.get_device_properties(0)\n",
        "start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)\n",
        "max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)\n",
        "print(f\"🖥️ Thiết bị GPU: {gpu_stats.name} | Dung lượng RAM tối đa: {max_memory} GB.\")\n",
        "print(f\"📊 VRAM đã chiếm dụng trước khi train: {start_gpu_memory} GB.\")"
    ]
})

# Cell 18: Run training (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Bắt đầu huấn luyện mô hình\n",
        "trainer_stats = trainer.train()"
    ]
})

# Cell 19: Save & inference title (Markdown)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Kiểm thử Mô hình & Lưu trữ LoRA Weights (Inference & Save)\n",
        "\n",
        "Sau khi train xong, chúng ta chạy thử dự đoán trên một ảnh X-ray y tế mẫu bất kỳ và lưu lại các file adapter LoRA đã được tối ưu."
    ]
})

# Cell 20: Inference (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import random\n",
        "\n",
        "# Chọn 1 mẫu bất kỳ trong dataset để kiểm tra\n",
        "test_sample = dataset[random.randint(0, len(dataset)-1)]\n",
        "image = test_sample[\"image\"].convert(\"RGB\")\n",
        "gold_caption = test_sample[\"caption\"]\n",
        "\n",
        "messages = [\n",
        "    {\n",
        "        \"role\": \"user\",\n",
        "        \"content\": [\n",
        "            {\"type\": \"image\"},\n",
        "            {\"type\": \"text\", \"text\": \"Describe what you see in this radiology image.\"}\n",
        "        ]\n",
        "    }\n",
        "]\n",
        "\n",
        "input_text = processor.apply_chat_template(messages, add_generation_prompt = True)\n",
        "inputs = processor(\n",
        "    image,\n",
        "    input_text,\n",
        "    add_special_tokens = False,\n",
        "    return_tensors = \"pt\",\n",
        ").to(\"cuda\")\n",
        "\n",
        "from transformers import TextStreamer\n",
        "text_streamer = TextStreamer(processor.tokenizer, skip_prompt = True)\n",
        "\n",
        "print(\"📝 Bệnh án gốc (Ground Truth):\")\n",
        "print(gold_caption)\n",
        "print(\"\\n📝 Mô tả dự đoán từ mô hình sau Fine-tuning:\")\n",
        "_ = model.generate(\n",
        "    **inputs, \n",
        "    streamer = text_streamer, \n",
        "    max_new_tokens = 128,\n",
        "    use_cache = True,\n",
        "    temperature = 1.0,\n",
        "    top_p = 0.95,\n",
        "    top_k = 64\n",
        ")"
    ]
})

# Cell 21: Save locally (Code)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Lưu adapter LoRA xuống thư mục cục bộ\n",
        "model.save_pretrained(\"gemma_4_lora\")\n",
        "processor.save_pretrained(\"gemma_4_lora\")\n",
        "print(\"✅ Lưu LoRA adapter thành công vào thư mục 'gemma_4_lora'!\")\n",
        "\n",
        "# Bạn có thể đẩy lên HuggingFace Hub bằng cách chạy code dưới:\n",
        "# model.push_to_hub(\"your_name/gemma_4_lora\", token = \"YOUR_HF_TOKEN\")\n",
        "# processor.push_to_hub(\"your_name/gemma_4_lora\", token = \"YOUR_HF_TOKEN\")"
    ]
})

# Construct notebook structure
notebook_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# Write notebook file
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=1, ensure_ascii=False)

print(f"✅ Generated Gemma-4 notebook successfully at: {notebook_path}")

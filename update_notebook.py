import json
import os

notebooks = [
    "Midterm/vqa_final/v3-vqa-2b-it.ipynb",
    "Midterm/vqa_final/v3-vqa-4b-it.ipynb"
]

cell_source = [
    "# ── BỔ SUNG: BENCHMARK TRÊN VQA-RAD TEST SPLIT (451 MẪU) ──────────────────────────\n",
    "# Cell này chạy độc lập để đánh giá mô hình vừa fine-tune trên tập test gốc của VQA-RAD.\n",
    "\n",
    "import re\n",
    "from datasets import load_dataset\n",
    "import pandas as pd\n",
    "from tqdm.auto import tqdm\n",
    "import torch\n",
    "\n",
    "# 1. Tải tập test VQA-RAD\n",
    "print(\"⏳ Đang tải flaviagiammarino/vqa-rad split=test...\")\n",
    "vqa_rad_test = load_dataset(\"flaviagiammarino/vqa-rad\", split=\"test\")\n",
    "print(f\"📊 VQA-RAD Test split: {len(vqa_rad_test)} mẫu\")\n",
    "\n",
    "# 2. Phân loại câu hỏi VQA-RAD thành 'open' hoặc 'closed' dựa trên câu trả lời\n",
    "vqa_rad_processed = []\n",
    "for sample in vqa_rad_test:\n",
    "    gold = str(sample[\"answer\"]).strip()\n",
    "    gold_lower = gold.lower()\n",
    "    qtype = \"closed\" if gold_lower in [\"yes\", \"no\"] else \"open\"\n",
    "    vqa_rad_processed.append({\n",
    "        \"image\": sample[\"image\"],\n",
    "        \"question\": sample[\"question\"],\n",
    "        \"question_type\": qtype,\n",
    "        \"gold\": gold,\n",
    "    })\n",
    "\n",
    "print(f\"✅ Đã chuẩn bị xong {len(vqa_rad_processed)} mẫu VQA-RAD.\")\n",
    "qtypes_cnt = {\"open\": 0, \"closed\": 0}\n",
    "for s in vqa_rad_processed:\n",
    "    qtypes_cnt[s[\"question_type\"]] += 1\n",
    "print(f\"   - Open-ended  : {qtypes_cnt['open']} mẫu\")\n",
    "print(f\"   - Closed-ended: {qtypes_cnt['closed']} mẫu\")\n",
    "\n",
    "\n",
    "# 3. Các hàm đánh giá và Inference tự chứa (self-contained)\n",
    "def local_normalize_answer(text: str) -> str:\n",
    "    text = str(text).lower().strip()\n",
    "    text = re.sub(r\"[^a-z0-9\\s]\", \" \", text)\n",
    "    text = re.sub(r\"\\s+\", \" \", text).strip()\n",
    "    return text\n",
    "\n",
    "def local_exact_match(pred: str, gold: str) -> int:\n",
    "    return int(local_normalize_answer(pred) == local_normalize_answer(gold))\n",
    "\n",
    "def local_token_f1(pred: str, gold: str) -> float:\n",
    "    p_toks = local_normalize_answer(pred).split()\n",
    "    g_toks = local_normalize_answer(gold).split()\n",
    "    if not p_toks or not g_toks:\n",
    "        return 0.0\n",
    "    common = set(p_toks) & set(g_toks)\n",
    "    if not common:\n",
    "        return 0.0\n",
    "    prec = sum(min(p_toks.count(t), g_toks.count(t)) for t in common) / len(p_toks)\n",
    "    rec  = sum(min(p_toks.count(t), g_toks.count(t)) for t in common) / len(g_toks)\n",
    "    if prec + rec == 0:\n",
    "        return 0.0\n",
    "    return 2 * prec * rec / (prec + rec)\n",
    "\n",
    "def local_run_inference(model, processor, sample: dict) -> str:\n",
    "    image = sample[\"image\"].convert(\"RGB\")\n",
    "    question_text = sample[\"question\"]\n",
    "\n",
    "    messages = [{\n",
    "        \"role\": \"user\",\n",
    "        \"content\": [\n",
    "            {\"type\": \"image\"},\n",
    "            {\"type\": \"text\", \"text\": question_text},\n",
    "        ],\n",
    "    }]\n",
    "\n",
    "    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n",
    "    inputs = processor(text=[text], images=[image], return_tensors=\"pt\", padding=True).to(model.device)\n",
    "\n",
    "    with torch.no_grad():\n",
    "        outputs = model.generate(\n",
    "            **inputs,\n",
    "            max_new_tokens=64,\n",
    "            do_sample=False,\n",
    "        )\n",
    "\n",
    "    input_len = inputs[\"input_ids\"].shape[1]\n",
    "    pred = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()\n",
    "    \n",
    "    pred_first_line = pred.strip().split(\"\\n\")[0].strip()\n",
    "    pred_short = \" \".join(pred_first_line.split()[:15])\n",
    "    return pred_short\n",
    "\n",
    "\n",
    "# 4. Hàm chạy Evaluation chính\n",
    "def run_evaluation_on_vqarad(model, processor):\n",
    "    from unsloth import FastModel\n",
    "    model = FastModel.for_inference(model)\n",
    "    \n",
    "    print(\"\\n🏃 Đang chạy inference trên 451 mẫu VQA-RAD...\")\n",
    "    rows = []\n",
    "    \n",
    "    for i, sample in enumerate(tqdm(vqa_rad_processed, desc=\"VQA-RAD Eval\")):\n",
    "        pred = local_run_inference(model, processor, sample)\n",
    "        rows.append({\n",
    "            \"idx\": i,\n",
    "            \"question_type\": sample[\"question_type\"],\n",
    "            \"question\": sample[\"question\"],\n",
    "            \"gold\": sample[\"gold\"],\n",
    "            \"pred\": pred,\n",
    "            \"em\": local_exact_match(pred, sample[\"gold\"]),\n",
    "            \"f1\": round(local_token_f1(pred, sample[\"gold\"]), 4),\n",
    "        })\n",
    "        \n",
    "    ems, f1s = [], []\n",
    "    by_type = {\"open\": {\"ems\": [], \"f1s\": []}, \"closed\": {\"ems\": [], \"f1s\": []}}\n",
    "    for r in rows:\n",
    "        em = r[\"em\"]\n",
    "        f1 = r[\"f1\"]\n",
    "        ems.append(em)\n",
    "        f1s.append(f1)\n",
    "        qt = r[\"question_type\"]\n",
    "        if qt in by_type:\n",
    "            by_type[qt][\"ems\"].append(em)\n",
    "            by_type[qt][\"f1s\"].append(f1)\n",
    "            \n",
    "    metrics = {\n",
    "        \"n\": len(rows),\n",
    "        \"exact_match\": round(sum(ems) / len(ems) * 100, 2),\n",
    "        \"token_f1\": round(sum(f1s) / len(f1s) * 100, 2),\n",
    "        \"em_open\": round(sum(by_type[\"open\"][\"ems\"]) / len(by_type[\"open\"][\"ems\"]) * 100, 2) if by_type[\"open\"][\"ems\"] else 0.0,\n",
    "        \"f1_open\": round(sum(by_type[\"open\"][\"f1s\"]) / len(by_type[\"open\"][\"f1s\"]) * 100, 2) if by_type[\"open\"][\"f1s\"] else 0.0,\n",
    "        \"em_closed\": round(sum(by_type[\"closed\"][\"ems\"]) / len(by_type[\"closed\"][\"ems\"]) * 100, 2) if by_type[\"closed\"][\"ems\"] else 0.0,\n",
    "        \"f1_closed\": round(sum(by_type[\"closed\"][\"f1s\"]) / len(by_type[\"closed\"][\"f1s\"]) * 100, 2) if by_type[\"closed\"][\"f1s\"] else 0.0,\n",
    "    }\n",
    "    \n",
    "    print(f\"\\n📊 KẾT QUẢ ĐÁNH GIÁ TRÊN VQA-RAD TEST SET:\")\n",
    "    print(f\"   - Tổng số mẫu: {metrics['n']}\")\n",
    "    print(f\"   - Overall EM : {metrics['exact_match']:.2f}%\")\n",
    "    print(f\"   - Overall F1 : {metrics['token_f1']:.2f}%\")\n",
    "    print(f\"   - EM (Open)   : {metrics['em_open']:.2f}%  |  F1 (Open)   : {metrics['f1_open']:.2f}%\")\n",
    "    print(f\"   - EM (Closed) : {metrics['em_closed']:.2f}%  |  F1 (Closed) : {metrics['f1_closed']:.2f}%\")\n",
    "    \n",
    "    try:\n",
    "        if \"DISCORD_WEBHOOK_URL\" in globals() and DISCORD_WEBHOOK_URL and DISCORD_WEBHOOK_URL != \"YOUR_DISCORD_WEBHOOK_URL\":\n",
    "            import requests\n",
    "            import datetime\n",
    "            timestamp = datetime.datetime.utcnow().isoformat() + \"Z\"\n",
    "            payload = {\n",
    "                \"username\": \"🤖 VQA Trainer Bot\",\n",
    "                \"embeds\": [{\n",
    "                    \"title\": \"🎉 ĐÁNH GIÁ MÔ HÌNH VỪA TRAIN TRÊN VQA-RAD TEST SET THÀNH CÔNG!\",\n",
    "                    \"description\": (\n",
    "                        f\"📊 **Overall EM**: `{metrics['exact_match']:.2f}%` · **Overall F1**: `{metrics['token_f1']:.2f}%`\\n\"\n",
    "                        f\"🔓 **EM (Open)**: `{metrics['em_open']:.2f}%` · **F1**: `{metrics['f1_open']:.2f}%`\\n\"\n",
    "                        f\"🔒 **EM (Closed)**: `{metrics['em_closed']:.2f}%` · **F1**: `{metrics['f1_closed']:.2f}%`\"\n",
    "                    ),\n",
    "                    \"color\": 0x00FF7F,\n",
    "                    \"footer\": {\"text\": f\"Completed at {timestamp[:19].replace('T', ' ')} UTC\"}\n",
                    "                }]\n",
    "            }\n",
    "            requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)\n",
    "            print(\"📨 Đã gửi kết quả đánh giá lên Discord webhook!\")\n",
    "    except Exception as e:\n",
    "        print(f\"⚠️ Không gửi được kết quả lên Discord: {e}\")\n",
    "        \n",
    "    df_results = pd.DataFrame(rows)\n",
    "    return df_results, metrics\n",
    "\n",
    "# Gọi hàm đánh giá\n",
    "df_vqa_rad_eval, metrics_vqa_rad_eval = run_evaluation_on_vqarad(model, processor)\n",
    "df_vqa_rad_eval.to_csv(\"vqa_rad_test_eval_results.csv\", index=False)\n"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {
        "trusted": True
    },
    "outputs": [],
    "source": cell_source
}

for nb_path in notebooks:
    if not os.path.exists(nb_path):
        print(f"⚠️ Không tìm thấy file: {nb_path}")
        continue
    
    print(f"⏳ Đang xử lý file: {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
    
    # Kiểm tra xem cell này đã tồn tại chưa bằng cách check chuỗi đặc trưng
    already_appended = False
    for cell in nb_data.get("cells", []):
        src = "".join(cell.get("source", []))
        if "local_run_inference" in src and "local_normalize_answer" in src:
            already_appended = True
            break
            
    if already_appended:
        print(f"ℹ️ File {nb_path} đã có cell này rồi — Bỏ qua.")
    else:
        nb_data["cells"].append(new_cell)
        print(f"✅ Đã thêm cell đánh giá vào file: {nb_path}")
        
    # Lưu lại file với định dạng đẹp (pretty-print)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, ensure_ascii=False, indent=1)
    print(f"💾 Đã lưu và format file: {nb_path}\n")

print("🎉 Hoàn tất cập nhật các notebook!")

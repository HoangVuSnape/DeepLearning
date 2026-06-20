import json

path = r"e:/DoCode/Master_Subject2026_TDTU/DeepLearning/Midterm/vqa_final/v3-vqa-2b-it.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Number of cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    src_preview = "".join(cell.get("source", []))[:100].replace("\n", " ")
    print(f"Cell {idx} ({cell['cell_type']}): {src_preview}")

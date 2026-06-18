import json

def inspect_notebook(path):
    print(f"=== Inspecting {path} ===")
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print("Number of cells:", len(nb.get("cells", [])))
    for idx, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get("cell_type")
        source = cell.get("source", [])
        source_str = "".join(source) if isinstance(source, list) else source
        first_lines = "\n".join(source_str.split("\n")[:3])
        print(f"Cell {idx} ({cell_type}):")
        print(first_lines)
        print("-" * 20)

inspect_notebook(r"e:\DoCode\Master_Subject2026_TDTU\DeepLearning\Midterm\code_ref\Gemma4_(E2B)_Vision.ipynb")
inspect_notebook(r"e:\DoCode\Master_Subject2026_TDTU\DeepLearning\Midterm\code_ref\Gemma4_(E4B)_Vision.ipynb")

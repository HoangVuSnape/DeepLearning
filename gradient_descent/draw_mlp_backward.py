"""Script to draw the specific MLP Backward Pass (Backpropagation) diagram using Matplotlib."""

import matplotlib.pyplot as plt
import numpy as np

def draw_network_backward():
    # Set up a wide, tall figure
    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=150)
    
    # Adjust subplot margins to leave the bottom 32% of the figure empty for text
    plt.subplots_adjust(bottom=0.32, top=0.9, left=0.05, right=0.95)
    
    # Layer positions (x-coordinates)
    layer_x = [1, 3.5, 6, 8.5]
    
    # Node positions (y-coordinates)
    nodes_l0 = [6.5, 4.5, 2.5]  # Input: Bias, x1, x2
    nodes_l1 = [6.5, 4.5, 2.5]  # H1: Bias, h1_1, h1_2
    nodes_l2 = [6.5, 4.5, 2.5]  # H2: Bias, h2_1, h2_2
    nodes_l3 = [4.5]             # Output: z
    
    # Draw Nodes
    # Layer 0 (Input)
    ax.scatter([layer_x[0]]*len(nodes_l0), nodes_l0, s=1500, color='#e9ecef', edgecolors='#6c757d', linewidths=2, zorder=5)
    labels_l0 = ["Bias\n1", "Input x1\n2", "Input x2\n3"]
    for y, label in zip(nodes_l0, labels_l0):
        ax.text(layer_x[0], y, label, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#495057', zorder=6)
        
    # Layer 1 (H1) with delta error values
    ax.scatter([layer_x[1]]*len(nodes_l1), nodes_l1, s=1500, color='#d4edda', edgecolors='#155724', linewidths=2, zorder=5)
    labels_l1 = [
        "Bias\n1", 
        "h1_1\nδ ≈ -0.0015", 
        "h1_2\nδ ≈ 0.0000"
    ]
    for y, label in zip(nodes_l1, labels_l1):
        ax.text(layer_x[1], y, label, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#155724', zorder=6)
        
    # Layer 2 (H2) with delta error values
    ax.scatter([layer_x[2]]*len(nodes_l2), nodes_l2, s=1500, color='#fff3cd', edgecolors='#856404', linewidths=2, zorder=5)
    labels_l2 = [
        "Bias\n1", 
        "h2_1\nδ ≈ -0.0087", 
        "h2_2\nδ ≈ 0.0012"
    ]
    for y, label in zip(nodes_l2, labels_l2):
        ax.text(layer_x[2], y, label, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#856404', zorder=6)
        
    # Layer 3 (Output) with delta error
    ax.scatter([layer_x[3]]*len(nodes_l3), nodes_l3, s=1500, color='#f8d7da', edgecolors='#721c24', linewidths=2, zorder=5)
    labels_l3 = ["Output z\nδ ≈ 0.0348\ny = 0.5"]
    for y, label in zip(nodes_l3, labels_l3):
        ax.text(layer_x[3], y, label, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#721c24', zorder=6)

    # Connections and backward red-orange dashed arrows showing error propagation
    # L2 <- L3 (Gradients & Deltas flow backwards)
    draw_backward_arrow(ax, layer_x[3], nodes_l3[0], layer_x[2], nodes_l2[0], "dw3_b = 0.0348", 0.22, '#dc3545')
    draw_backward_arrow(ax, layer_x[3], nodes_l3[0], layer_x[2], nodes_l2[1], "dw3_1 = 0.0297", 0.28, '#dc3545')
    draw_backward_arrow(ax, layer_x[3], nodes_l3[0], layer_x[2], nodes_l2[2], "dw3_2 = 0.0335", 0.28, '#dc3545')

    # L1 <- L2 (Gradients & Deltas flow backwards)
    draw_backward_arrow(ax, layer_x[2], nodes_l2[1], layer_x[1], nodes_l1[0], "dw2_b1 = -0.0087", 0.18, '#fd7e14')
    draw_backward_arrow(ax, layer_x[2], nodes_l2[2], layer_x[1], nodes_l1[0], "dw2_b2 = 0.0012", 0.12, '#fd7e14')
    
    draw_backward_arrow(ax, layer_x[2], nodes_l2[1], layer_x[1], nodes_l1[1], "dw2_11 = -0.0023", 0.22, '#fd7e14')
    draw_backward_arrow(ax, layer_x[2], nodes_l2[2], layer_x[1], nodes_l1[1], "dw2_21 = 0.0003", 0.32, '#fd7e14')
    
    draw_backward_arrow(ax, layer_x[2], nodes_l2[1], layer_x[1], nodes_l1[2], "dw2_12 = -0.0087", 0.32, '#fd7e14')
    draw_backward_arrow(ax, layer_x[2], nodes_l2[2], layer_x[1], nodes_l1[2], "dw2_22 = 0.0012", 0.22, '#fd7e14')

    # L0 <- L1 (Gradients & Deltas flow backwards)
    draw_backward_arrow(ax, layer_x[1], nodes_l1[1], layer_x[0], nodes_l0[0], "dw1_b1 = -0.0015", 0.18, '#6c757d')
    draw_backward_arrow(ax, layer_x[1], nodes_l1[2], layer_x[0], nodes_l0[0], "dw1_b2 ≈ 0", 0.12, '#6c757d')
    
    draw_backward_arrow(ax, layer_x[1], nodes_l1[1], layer_x[0], nodes_l0[1], "dw1_11 = -0.0029", 0.22, '#6c757d')
    draw_backward_arrow(ax, layer_x[1], nodes_l1[2], layer_x[0], nodes_l0[1], "dw1_21 ≈ 0", 0.32, '#6c757d')
    
    draw_backward_arrow(ax, layer_x[1], nodes_l1[1], layer_x[0], nodes_l0[2], "dw1_12 = -0.0044", 0.32, '#6c757d')
    draw_backward_arrow(ax, layer_x[1], nodes_l1[2], layer_x[0], nodes_l0[2], "dw1_22 ≈ 0", 0.22, '#6c757d')

    # Layer Headers
    ax.text(layer_x[0], 7.5, "Lớp Đầu Vào\n(Input Layer)", ha='center', va='center', fontsize=12, fontweight='bold', color='#495057')
    ax.text(layer_x[1], 7.5, "Lớp Ẩn H1\n(Hidden Layer 1)", ha='center', va='center', fontsize=12, fontweight='bold', color='#155724')
    ax.text(layer_x[2], 7.5, "Lớp Ẩn H2\n(Hidden Layer 2)", ha='center', va='center', fontsize=12, fontweight='bold', color='#856404')
    ax.text(layer_x[3], 7.5, "Lớp Đầu Ra\n(Output Layer)", ha='center', va='center', fontsize=12, fontweight='bold', color='#721c24')

    # Set boundaries and details
    ax.set_xlim(0, 9.5)
    ax.set_ylim(1.8, 8.0)
    ax.set_title("Sơ Đồ Lan Truyền Ngược (Backpropagation) - Tính Toán Lỗi & Gradient", fontsize=15, fontweight='bold', pad=25)
    
    # Subtitles/Explanations placed completely outside the network drawing area
    explanation = (
        "Quy trình tính toán lan truyền ngược (Backward Pass) với y = 0.5:\n"
        "1. Lớp đầu ra (Layer 3):\n"
        "   - δ(output) = (ŷ - y) * ŷ(1 - ŷ) = (0.9051 - 0.5) * 0.9051 * (1 - 0.9051) ≈ 0.0348\n"
        "   - dL/dw3 = δ(output) * a(H2)  ==>  dw3_1 ≈ 0.0297, dw3_2 ≈ 0.0335, dLoss/db3 ≈ 0.0348\n"
        "2. Lớp ẩn H2 (Layer 2):\n"
        "   - δ(h2_1) = δ(output) * w3_1 * σ'(z2_1) = (0.0348 * -2) * (0.8538 * 0.1462) ≈ -0.0087\n"
        "   - δ(h2_2) = δ(output) * w3_2 * σ'(z2_2) = (0.0348 * 1) * (0.9631 * 0.0369) ≈ 0.0012\n"
        "   - dL/dw2 = δ(H2) * a(H1)      ==>  dw2_11 ≈ -0.0023, dw2_12 ≈ -0.0087, dw2_21 ≈ 0.0003, dw2_22 ≈ 0.0012\n"
        "3. Lớp ẩn H1 (Layer 1):\n"
        "   - δ(h1_1) = [δ(h2_1)*w2_11 + δ(h2_2)*w2_21] * σ'(z1_1) ≈ -0.0015\n"
        "   - δ(h1_2) = [δ(h2_1)*w2_12 + δ(h2_2)*w2_22] * σ'(z1_2) ≈ -0.000025 (≈ 0.0000)"
    )
    
    # Place text box at the bottom center of the figure with ample padding (no monospace to prevent glyph warnings)
    props = dict(boxstyle='round,pad=1.2', facecolor='#f8f9fa', edgecolor='#ced4da', alpha=0.95)
    fig.text(0.08, 0.03, explanation, fontsize=10.5, bbox=props, va='bottom')

    ax.axis('off')
    plt.savefig("docs/6.4_Backpropagation/mlp_backpropagation_flow.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Sơ đồ backward đã được lưu tại: docs/6.4_Backpropagation/mlp_backpropagation_flow.png")

def draw_backward_arrow(ax, x1, y1, x2, y2, label, label_pos=0.5, color='red'):
    # Adjust coordinates to not enter the node circles (radius ≈ 0.35 in data coordinates)
    # The arrow goes from right to left: from (x1, y1) to (x2, y2)
    node_radius = 0.35
    angle = np.arctan2(y2 - y1, x2 - x1)
    
    start_x = x1 + node_radius * np.cos(angle)
    start_y = y1 + node_radius * np.sin(angle)
    end_x = x2 - node_radius * np.cos(angle)
    end_y = y2 - node_radius * np.sin(angle)
    
    # Draw dashed backward arrow
    ax.annotate("",
                xy=(end_x, end_y), xycoords='data',
                xytext=(start_x, start_y), textcoords='data',
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, ls="--", shrinkA=0, shrinkB=0, alpha=0.75))
    
    # Text label position
    lx = start_x + (end_x - start_x) * label_pos
    ly = start_y + (end_y - start_y) * label_pos
    
    # Offset label to not sit exactly on the line
    offset = 0.14
    dx = end_x - start_x
    dy = end_y - start_y
    length = np.sqrt(dx**2 + dy**2)
    nx = -dy / length * offset
    ny = dx / length * offset
    
    ax.text(lx + nx, ly + ny, label, color=color, fontsize=8.5, fontweight='bold', ha='center', va='center')

if __name__ == "__main__":
    draw_network_backward()

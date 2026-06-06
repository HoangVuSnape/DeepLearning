"""Script to draw the specific MLP Forward Pass diagram from the class slide using Matplotlib."""

import matplotlib.pyplot as plt
import numpy as np

def draw_network():
    # Set up a wide, tall figure
    fig, ax = plt.subplots(figsize=(13, 8.5), dpi=150)
    
    # Adjust subplot margins to leave the bottom 30% of the figure empty for text
    plt.subplots_adjust(bottom=0.32, top=0.9, left=0.05, right=0.95)
    
    # Layer positions (x-coordinates)
    layer_x = [1, 3.5, 6, 8.5]
    
    # Node positions (y-coordinates) - spaced out vertically
    nodes_l0 = [6.5, 4.5, 2.5]  # Input: Bias, x1, x2
    nodes_l1 = [6.5, 4.5, 2.5]  # H1: Bias, h1_1, h1_2
    nodes_l2 = [6.5, 4.5, 2.5]  # H2: Bias, h2_1, h2_2
    nodes_l3 = [4.5]             # Output: z
    
    # Draw Nodes
    # Layer 0 (Input)
    ax.scatter([layer_x[0]]*len(nodes_l0), nodes_l0, s=1500, color='#d1ecf1', edgecolors='#0c5460', linewidths=2, zorder=5)
    labels_l0 = ["Bias\n1", "Input x1\n2", "Input x2\n3"]
    for y, label in zip(nodes_l0, labels_l0):
        ax.text(layer_x[0], y, label, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0c5460', zorder=6)
        
    # Layer 1 (H1)
    ax.scatter([layer_x[1]]*len(nodes_l1), nodes_l1, s=1500, color='#d4edda', edgecolors='#155724', linewidths=2, zorder=5)
    labels_l1 = ["Bias\n1", "h1_1\n(Σ/σ)", "h1_2\n(Σ/σ)"]
    for y, label in zip(nodes_l1, labels_l1):
        ax.text(layer_x[1], y, label, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#155724', zorder=6)
        
    # Layer 2 (H2)
    ax.scatter([layer_x[2]]*len(nodes_l2), nodes_l2, s=1500, color='#fff3cd', edgecolors='#856404', linewidths=2, zorder=5)
    labels_l2 = ["Bias\n1", "h2_1\n(Σ/σ)", "h2_2\n(Σ/σ)"]
    for y, label in zip(nodes_l2, labels_l2):
        ax.text(layer_x[2], y, label, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#856404', zorder=6)
        
    # Layer 3 (Output)
    ax.scatter([layer_x[3]]*len(nodes_l3), nodes_l3, s=1500, color='#f8d7da', edgecolors='#721c24', linewidths=2, zorder=5)
    labels_l3 = ["Output z\n(Σ/σ)\ny = ?"]
    for y, label in zip(nodes_l3, labels_l3):
        ax.text(layer_x[3], y, label, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#721c24', zorder=6)

    # Connections and weight labels
    # Connections L0 -> L1
    # Node 1 of L0 (Bias) to H1_1 (weight 1), H1_2 (weight 2)
    draw_arrow(ax, layer_x[0], nodes_l0[0], layer_x[1], nodes_l1[1], "w = 1", 0.18, '#0c5460')
    draw_arrow(ax, layer_x[0], nodes_l0[0], layer_x[1], nodes_l1[2], "w = 2", 0.12, '#0c5460')
    # Node 2 of L0 (x1) to H1_1 (weight -4), H1_2 (weight 1)
    draw_arrow(ax, layer_x[0], nodes_l0[1], layer_x[1], nodes_l1[1], "w = -4", 0.22, '#0c5460')
    draw_arrow(ax, layer_x[0], nodes_l0[1], layer_x[1], nodes_l1[2], "w = 1", 0.32, '#0c5460')
    # Node 3 of L0 (x2) to H1_1 (weight 2), H1_2 (weight 0.5)
    draw_arrow(ax, layer_x[0], nodes_l0[2], layer_x[1], nodes_l1[1], "w = 2", 0.32, '#0c5460')
    draw_arrow(ax, layer_x[0], nodes_l0[2], layer_x[1], nodes_l1[2], "w = 0.5", 0.22, '#0c5460')

    # Connections L1 -> L2
    # Node 1 of L1 (Bias) to H2_1 (weight 0.5), H2_2 (weight 1)
    draw_arrow(ax, layer_x[1], nodes_l1[0], layer_x[2], nodes_l2[1], "w = 0.5", 0.18, '#155724')
    draw_arrow(ax, layer_x[1], nodes_l1[0], layer_x[2], nodes_l2[2], "w = 1", 0.12, '#155724')
    # Node 2 of L1 (h1_1) to H2_1 (weight 1), H2_2 (weight 1)
    draw_arrow(ax, layer_x[1], nodes_l1[1], layer_x[2], nodes_l2[1], "w = 1", 0.22, '#155724')
    draw_arrow(ax, layer_x[1], nodes_l1[1], layer_x[2], nodes_l2[2], "w = 1", 0.32, '#155724')
    # Node 3 of L1 (h1_2) to H2_1 (weight 1), H2_2 (weight 2)
    draw_arrow(ax, layer_x[1], nodes_l1[2], layer_x[2], nodes_l2[1], "w = 1", 0.32, '#155724')
    draw_arrow(ax, layer_x[1], nodes_l1[2], layer_x[2], nodes_l2[2], "w = 2", 0.22, '#155724')

    # Connections L2 -> L3
    # Node 1 of L2 (Bias) to Output (weight 3)
    draw_arrow(ax, layer_x[2], nodes_l2[0], layer_x[3], nodes_l3[0], "w = 3", 0.2, '#856404')
    # Node 2 of L2 (h2_1) to Output (weight -2)
    draw_arrow(ax, layer_x[2], nodes_l2[1], layer_x[3], nodes_l3[0], "w = -2", 0.25, '#856404')
    # Node 3 of L2 (h2_2) to Output (weight 1)
    draw_arrow(ax, layer_x[2], nodes_l2[2], layer_x[3], nodes_l3[0], "w = 1", 0.25, '#856404')

    # Layer Headers
    ax.text(layer_x[0], 7.5, "Lớp Đầu Vào\n(Input Layer)", ha='center', va='center', fontsize=12, fontweight='bold', color='#0c5460')
    ax.text(layer_x[1], 7.5, "Lớp Ẩn H1\n(Hidden Layer 1)", ha='center', va='center', fontsize=12, fontweight='bold', color='#155724')
    ax.text(layer_x[2], 7.5, "Lớp Ẩn H2\n(Hidden Layer 2)", ha='center', va='center', fontsize=12, fontweight='bold', color='#856404')
    ax.text(layer_x[3], 7.5, "Lớp Đầu Ra\n(Output Layer)", ha='center', va='center', fontsize=12, fontweight='bold', color='#721c24')

    # Set boundaries and details
    ax.set_xlim(0, 9.5)
    ax.set_ylim(1.8, 8.0)
    ax.set_title("Sơ Đồ Forward Pass MLP Tính Toán Bằng Số Chi Tiết", fontsize=15, fontweight='bold', pad=25)
    
    # Subtitles/Explanations placed completely outside the network drawing area
    explanation = (
        "Quy trình tính toán từng nút (Forward Pass):\n"
        "1. Lớp ẩn H1:\n"
        "   - z(h1_1) = 1*1 + 2*(-4) + 3*2 = -1  ==>  a(h1_1) = σ(-1) = 1 / (1 + e^1) ≈ 0.2689\n"
        "   - z(h1_2) = 1*2 + 2*1 + 3*0.5 = 5.5  ==>  a(h1_2) = σ(5.5) = 1 / (1 + e^-5.5) ≈ 0.9959\n"
        "2. Lớp ẩn H2:\n"
        "   - z(h2_1) = 1*0.5 + a(h1_1)*1 + a(h1_2)*1 ≈ 1.7648  ==>  a(h2_1) = σ(1.7648) ≈ 0.8538\n"
        "   - z(h2_2) = 1*1 + a(h1_1)*1 + a(h1_2)*2 ≈ 3.2608  ==>  a(h2_2) = σ(3.2608) ≈ 0.9631\n"
        "3. Lớp đầu ra (Output):\n"
        "   - z(output) = 1*3 + a(h2_1)*(-2) + a(h2_2)*1 ≈ 2.2554\n"
        "   - y = a(output) = σ(2.2554) = 1 / (1 + e^-2.2554) ≈ 0.9051"
    )
    
    # Place text box at the bottom center of the figure with ample padding
    props = dict(boxstyle='round,pad=1.2', facecolor='#f8f9fa', edgecolor='#ced4da', alpha=0.95)
    fig.text(0.08, 0.03, explanation, fontsize=10.5, bbox=props, va='bottom')

    ax.axis('off')
    plt.savefig("docs/6.4_Backpropagation/mlp_slide_forward.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Sơ đồ đã được cập nhật và lưu tại: docs/6.4_Backpropagation/mlp_slide_forward.png")

def draw_arrow(ax, x1, y1, x2, y2, label, label_pos=0.5, color='gray'):
    # Adjust coordinates to not enter the node circles (radius ≈ 0.32 in data coordinates)
    node_radius = 0.35
    angle = np.arctan2(y2 - y1, x2 - x1)
    
    start_x = x1 + node_radius * np.cos(angle)
    start_y = y1 + node_radius * np.sin(angle)
    end_x = x2 - node_radius * np.cos(angle)
    end_y = y2 - node_radius * np.sin(angle)
    
    # Draw arrow line
    ax.annotate("",
                xy=(end_x, end_y), xycoords='data',
                xytext=(start_x, start_y), textcoords='data',
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, shrinkA=0, shrinkB=0, alpha=0.7))
    
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
    
    ax.text(lx + nx, ly + ny, label, color=color, fontsize=9, fontweight='bold', ha='center', va='center')

if __name__ == "__main__":
    draw_network()

# Bảng tra cứu thuật ngữ & Ký hiệu Toán học (Glossary) trong Backpropagation

Tài liệu này hệ thống hóa toàn bộ các ký hiệu, khái niệm, kiểu dữ liệu (Scalar, Vector, Matrix) và kích thước (Dimensions) của các biến số được sử dụng trong thuật toán Lan truyền ngược (Backpropagation) của mạng MLP (Multi-Layer Perceptron).

---

## 1. Định nghĩa kích thước mạng (Network Dimensions)
Trước khi đi vào chi tiết các biến, ta quy ước kích thước của mạng nơ-ron như sau:
* **$L$**: Tổng số lớp tính toán của mạng (không tính lớp đầu vào $0$).
* **$l$**: Chỉ số đại diện cho lớp hiện tại ($l \in \{1, 2, \dots, L\}$).
* **$n_l$**: Số lượng nơ-ron (nodes) tại lớp $l$.
* **$n_{l-1}$**: Số lượng nơ-ron tại lớp liền trước $l-1$.
* **$n_0$**: Số lượng đặc trưng đầu vào (input features).

---

## 2. Bảng tra cứu chi tiết các biến số (Glossary Table)

| Ký hiệu | Tên gọi tiếng Việt | Tên gọi tiếng Anh | Kiểu dữ liệu | Kích thước (Dimensions) | Ý nghĩa / Công thức |
| :--- | :--- | :--- | :---: | :---: | :--- |
| $L$ | Hàm mất mát / Hàm lỗi | Loss function | **Scalar** | $1 \times 1$ | Đo lường độ sai lệch giữa dự đoán và thực tế. Ví dụ MSE: $L = \frac{1}{2}\sum (y_i - a_i^{[L]})^2$. |
| $\eta$ | Tốc độ học | Learning rate | **Scalar** | $1 \times 1$ | Hệ số điều chỉnh độ lớn của bước cập nhật tham số. |
| $x$ | Vector đầu vào | Input vector | **Vector** | $n_0 \times 1$ | Dữ liệu đầu vào của mô hình. Tương đương với $a^{[0]}$. |
| $y$ | Vector nhãn thực tế | Target vector | **Vector** | $n_L \times 1$ | Nhãn thực tế (ground truth) của dữ liệu. |
| $z_i^{[l]}$ | Giá trị tiền kích hoạt của node $i$ | Pre-activation of node $i$ | **Scalar** | $1 \times 1$ | Tổng tuyến tính tại node $i$ lớp $l$: $z_i^{[l]} = \sum (W_{ij}^{[l]} a_j^{[l-1]}) + b_i^{[l]}$. |
| $z^{[l]}$ | Vector tiền kích hoạt lớp $l$ | Pre-activation vector | **Vector** | $n_l \times 1$ | Tập hợp tất cả $z_i^{[l]}$ của lớp $l$: $z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$. |
| $a_i^{[l]}$ | Giá trị kích hoạt của node $i$ | Activation of node $i$ | **Scalar** | $1 \times 1$ | Đầu ra phi tuyến tại node $i$ lớp $l$: $a_i^{[l]} = g^{[l]}(z_i^{[l]})$. |
| $a^{[l]}$ | Vector kích hoạt lớp $l$ | Activation vector | **Vector** | $n_l \times 1$ | Đầu ra của lớp $l$ sau hàm kích hoạt: $a^{[l]} = g^{[l]}(z^{[l]})$. |
| $g^{[l]}$ | Hàm kích hoạt lớp $l$ | Activation function | **Hàm số** | - | Hàm phi tuyến như Sigmoid, ReLU, Tanh, Softmax. |
| $g^{[l]\prime}$ | Đạo hàm hàm kích hoạt | Derivative of activation | **Hàm số** | - | Đạo hàm của $g^{[l]}$ theo biến $z$. |
| $W_{ij}^{[l]}$ | Trọng số liên kết | Connection weight | **Scalar** | $1 \times 1$ | Trọng số từ nơ-ron $j$ (lớp $l-1$) tới nơ-ron $i$ (lớp $l$). |
| $W^{[l]}$ | Ma trận trọng số lớp $l$ | Weight matrix | **Matrix** | $n_l \times n_{l-1}$ | Ma trận chứa toàn bộ trọng số kết nối giữa lớp $l-1$ và lớp $l$. |
| $b_i^{[l]}$ | Bias của node $i$ lớp $l$ | Bias of node $i$ | **Scalar** | $1 \times 1$ | Giá trị chệch của nơ-ron $i$ ở lớp $l$. |
| $b^{[l]}$ | Vector bias lớp $l$ | Bias vector | **Vector** | $n_l \times 1$ | Vector chứa toàn bộ bias của lớp $l$. |
| $\delta_i^{[l]}$ | Sai số / Độ lỗi tại node $i$ | Error term (Delta) of node $i$ | **Scalar** | $1 \times 1$ | Đạo hàm riêng của Loss theo tiền kích hoạt: $\delta_i^{[l]} = \frac{\partial L}{\partial z_i^{[l]}}$. |
| $\delta^{[l]}$ | Vector sai số lớp $l$ | Error vector (Delta) | **Vector** | $n_l \times 1$ | Tập hợp sai số của tất cả các node tại lớp $l$. |
| $\frac{\partial L}{\partial W^{[l]}}$ | Gradient của ma trận trọng số | Gradient of Weight matrix | **Matrix** | $n_l \times n_{l-1}$ | Các đạo hàm riêng dùng để cập nhật $W^{[l]}$: $\frac{\partial L}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^T$. |
| $\frac{\partial L}{\partial b^{[l]}}$ | Gradient của vector bias | Gradient of bias vector | **Vector** | $n_l \times 1$ | Các đạo hàm riêng dùng để cập nhật $b^{[l]}$: $\frac{\partial L}{\partial b^{[l]}} = \delta^{[l]}$. |

---

## 3. Các phép toán đặc biệt
* **Nhân ma trận thông thường ($\times$ hoặc viết liền)**: 
  * Phép nhân ma trận $A$ kích thước $(m \times n)$ với ma trận $B$ kích thước $(n \times p)$ tạo ra ma trận $(m \times p)$.
  * Ví dụ: $W^{[l]} a^{[l-1]} \rightarrow (n_l \times n_{l-1}) \times (n_{l-1} \times 1) = (n_l \times 1)$.
* **Phép chuyển vị (Transpose - ký hiệu $T$)**:
  * Đổi các dòng thành các cột và ngược lại.
  * Ví dụ: $(W^{[l]})^T$ có kích thước từ $(n_l \times n_{l-1})$ thành $(n_{l-1} \times n_l)$.
* **Nhân Hadamard (Element-wise product - ký hiệu $\odot$)**:
  * Nhân từng phần tử tương ứng của hai ma trận hoặc hai vector có cùng kích thước.
  * Ví dụ: $A \odot B$ yêu cầu $A$ và $B$ phải có cùng kích thước (ví dụ đều là $n_l \times 1$). Kết quả nhận được cũng có kích thước $n_l \times 1$.

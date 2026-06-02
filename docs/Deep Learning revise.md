---
created: 2026-05-30T15:28:00
updated:
  - 2026-05-30T16:14:15+07:00
  - 2026-05-30T16:13:35+07:00
  - 2026-05-30T16:12:04+07:00
  - 2026-05-30T16:09:39+07:00
  - 2026-05-30T16:08:13+07:00
  - 2026-05-30T16:07:17+07:00
  - 2026-05-30T16:05:42+07:00
  - 2026-05-30T15:37:57+07:00
  - 2026-05-30T15:30:30+07:00
  - 2026-05-30T15:29:08+07:00
  - 2026-05-30T15:24:17+07:00
  - 2026-05-30T14:38:50+07:00
  - 2026-05-30T14:15:50+07:00
  - 2026-05-30T14:08:47+07:00
  - 2026-05-30T14:06:08+07:00
  - 2026-05-30T14:05:36+07:00
  - 2026-05-30T14:03:56+07:00
  - 2026-05-30T13:59:04+07:00
  - 2026-05-30T13:58:20+07:00
  - 2026-05-30T13:57:34+07:00
  - 2026-05-30T13:50:54+07:00
  - 2026-05-30T13:49:42+07:00
  - 2026-05-30T13:48:40+07:00
  - 2026-05-30T13:47:11+07:00
  - 2026-05-30T13:40:31+07:00
  - 2026-05-30T13:39:04+07:00
  - 2026-05-30T13:38:16+07:00
  - 2026-05-30T13:32:14+07:00
  - 2026-05-30T13:31:00+07:00
  - 2026-05-30T13:29:57+07:00
  - 2026-05-30T13:27:30+07:00
  - 2026-05-30T13:25:17+07:00
  - 2026-05-30T13:12:27+07:00
  - 2025-11-24T13:52:00+07:00
  - <% tp.file.last_modified_date("YYYY-MM-DD HH:mm:ss") %>
dg-publish:
dg-home:
tags:
dg-pinned: "false"
aliases:
cssclasses:
---
# Buổi 1 - Tổng quan mạng nơ-ron (Deep Learning)

## Mục tiêu
- Hiểu cấu trúc mạng nơ-ron: input -> hidden -> output.
- Hiểu cơ chế học: forward -> loss -> backprop -> gradient descent.
- Hiểu vai trò của weights, bias, activation.

## Cấu trúc một node
1. Bước tuyến tính:
   $$z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$
2. Bước phi tuyến:
   $$a = f(z)$$

Trong đó $w$ điều chỉnh độ quan trọng của từng đầu vào, $b$ cho phép dịch chuyển ngưỡng kích hoạt, và $f$ tạo độ cong (phi tuyến).

## 1. Lan truyền tiến (Forward propagation)
- Input vào từng lớp ẩn.
- Mỗi node tính $z$ (tổng có trọng số) và qua hàm kích hoạt $f$.
- Lớp output tạo ra dự đoán $\hat{y}$.

**Loss/Error:** so sánh $\hat{y}$ với $y$ để tính sai số.

## 2. Lan truyền ngược (Backpropagation)
- Dùng quy tắc chuỗi để lan truyền sai số từ output về các lớp trước.
- Tính gradient cho từng $w, b$.

## 3. Gradient Descent (cập nhật tham số)
$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$$
$$b_{new} = b_{old} - \eta \cdot \frac{\partial L}{\partial b}$$

Trong đó $\eta$ là learning rate.

---

## Vì sao activation phải phi tuyến?
- Nếu tất cả các lớp đều tuyến tính, toàn bộ mạng sẽ rút gọn thành một lớp tuyến tính duy nhất.
- Hàm phi tuyến giúp mạng học các biến đổi phức tạp và ranh giới cong.
- Đạo hàm không phải hằng số -> cập nhật gradient có ý nghĩa.

## Vai trò của $w$, $b$, và activation
- **Weights ($w$):** điều chỉnh độ quan trọng của từng feature.
- **Bias ($b$):** dịch chuyển ngưỡng kích hoạt, tránh bị ép qua gốc tọa độ.
- **Activation ($f$):** tạo phi tuyến, giúp mô hình biểu diễn dữ liệu phức tạp.

---

## Hàm Sigmoid (ví dụ activation)
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- Giá trị trong khoảng $(0, 1)$.
- Hay dùng cho bài toán phân loại nhị phân.
- Đạo hàm:
  $$\sigma'(z) = \sigma(z) (1 - \sigma(z))$$

---

## Softmax và Cross-Entropy (phân loại nhiều nhãn)
- Softmax biến logit thành phân phối xác suất:
  $$\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
- Cross-entropy cho 1 sample:
  $$L = -\sum_i y_i \log(\hat{y}_i)$$

Ví dụ (1 sample):
- $y = [0, 1, 0]$, $\hat{y} = [0.3, 0.6, 0.1]$
- $L = -\log(0.6) = 0.5108$

---

# Ghi chú / QA
- Tại sao activation phải phi tuyến?
- Tại sao trong 1 node cần đủ $w$, $b$, và activation?

# Bài tập / Đề 2
- Classification, Softmax, 3 nhãn.

# Tính toán
- [Calcuate](Calcuate.md)

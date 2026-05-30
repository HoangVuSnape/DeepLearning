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
# Buoi 1 - Tong quan mang no-ron (Deep Learning)

## Muc tieu
- Hieu cau truc mang no-ron: input -> hidden -> output.
- Hieu co che hoc: forward -> loss -> backprop -> gradient descent.
- Hieu vai tro cua weights, bias, activation.

## Cau truc mot node
1. Buoc tuyen tinh:
   $$z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$
2. Buoc phi tuyen:
   $$a = f(z)$$

Trong do $w$ dieu chinh do quan trong cua tung dau vao, $b$ cho phep dich chuyen nguong kich hoat, va $f$ tao do cong (phi tuyen).

## 1. Lan truyen tien (Forward propagation)
- Input vao tung lop an.
- Moi node tinh $z$ (tong co trong so) va qua ham kich hoat $f$.
- Lop output tao ra du doan $\hat{y}$.

**Loss/Error:** so sanh $\hat{y}$ voi $y$ de tinh sai so.

## 2. Lan truyen nguoc (Backpropagation)
- Dung quy tac chuoi de lan truyen sai so tu output ve cac lop truoc.
- Tinh gradient cho tung $w, b$.

## 3. Gradient Descent (cap nhat tham so)
$$w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w}$$
$$b_{new} = b_{old} - \eta \cdot \frac{\partial L}{\partial b}$$

Trong do $\eta$ la learning rate.

---

## Vi sao activation phai phi tuyen?
- Neu tat ca cac lop deu tuyen tinh, toan bo mang se rut gon thanh mot lop tuyen tinh duy nhat.
- Ham phi tuyen giup mang hoc cac bien doi phuc tap va ranh gioi cong.
- Dao ham khong phai hang so -> cap nhat gradient co y nghia.

## Vai tro cua $w$, $b$, va activation
- **Weights ($w$):** dieu chinh do quan trong cua tung feature.
- **Bias ($b$):** dich chuyen nguong kich hoat, tranh bi ep qua goc toan do.
- **Activation ($f$):** tao phi tuyen, giup mo hinh bieu dien du lieu phuc tap.

---

## Ham Sigmoid (vi du activation)
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- Gia tri trong khoang $(0, 1)$.
- Hay dung cho bai toan phan loai nhi phan.
- Dao ham:
  $$\sigma'(z) = \sigma(z) (1 - \sigma(z))$$

---

## Softmax va Cross-Entropy (phan loai nhieu nhan)
- Softmax bien logit thanh phan phoi xac suat:
  $$\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
- Cross-entropy cho 1 sample:
  $$L = -\sum_i y_i \log(\hat{y}_i)$$

Vi du (1 sample):
- $y = [0, 1, 0]$, $\hat{y} = [0.3, 0.6, 0.1]$
- $L = -\log(0.6) = 0.5108$

---

# Ghi chu / QA
- Tai sao activation phai phi tuyen?
- Tai sao trong 1 node can du $w$, $b$, va activation?

# Bai tap / De 2
- Classification, Softmax, 3 nhan.

# Tinh toan
- [Calcuate](Calcuate.md)

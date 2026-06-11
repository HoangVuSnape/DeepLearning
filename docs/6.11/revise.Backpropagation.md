Dưới đây là nội dung và các công thức chi tiết được trích xuất từ hình ảnh của bạn về thuật toán lan truyền ngược (backpropagation):

### Các bước suy ra công thức sai số lớp ẩn

**1. Bắt đầu từ định nghĩa sai số** (theo chuỗi phụ thuộc $z^l \rightarrow a^l \rightarrow z^{l+1} \rightarrow \dots \rightarrow L$)

* Định nghĩa sai số tại lớp $l$:

$$\delta^l = \frac{\partial L}{\partial z^l}$$


* Phương trình tính toán tại lớp $l$:

$$z^l = W^l a^{l-1} + b^l$$



**2. Tách đạo hàm qua $a^l$**, dùng $\frac{\partial a^l}{\partial z^l} = g'(z^l)$

* Áp dụng quy tắc chuỗi (chain rule):

$$\delta^l = \frac{\partial L}{\partial a^l} \odot g'(z^l)$$


* Hàm kích hoạt tại lớp $l$:

$$a^l = g^l(z^l)$$



**3. Tính $\frac{\partial L}{\partial a^l}$ qua lớp $l+1$**, với $\delta^{l+1} = \frac{\partial L}{\partial z^{l+1}}$

* Đạo hàm của hàm mất mát theo giá trị kích hoạt:

$$\frac{\partial L}{\partial a^l} = (W^{l+1})^T \delta^{l+1}$$



**4. Thay ngược vào bước 2 $\rightarrow$ công thức lan truyền ngược sai số**

* Công thức tổng quát cho sai số tại lớp ẩn $l$:

$$\delta^l = (W^{l+1})^T \delta^{l+1} \odot g'(z^l)$$



*(Lưu ý: Ký hiệu $\odot$ đại diện cho phép nhân Hadamard - nhân từng phần tử của hai ma trận hoặc vector).*
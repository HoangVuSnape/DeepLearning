# Cheat Sheet: Ký hiệu & Ý nghĩa toán học trong Tối ưu hóa

Bảng tra cứu nhanh các ký hiệu toán học phổ biến được sử dụng xuyên suốt trong các công thức của các bộ tối ưu hóa (Optimizers):

| Ký hiệu | Tên gọi (Tiếng Việt) | English Term | Vai trò / Ý nghĩa toán học | Giá trị mặc định phổ biến |
| :---: | :--- | :--- | :--- | :---: |
| $\theta$ | Tham số mô hình | Model Parameters | Bao gồm toàn bộ trọng số ($W$) và độ chệch ($b$) cần tối ưu hóa: $\theta = \{W, b\}$. | - |
| $t$ | Bước thời gian (bước lặp) | Timestep / Iteration | Chỉ số bước cập nhật hiện tại (bắt đầu từ $t = 1$). | - |
| $L(\theta)$ | Hàm mất mát | Loss Function | Hàm số đo lường sai số của mô hình. Mục tiêu là cực tiểu hóa $L(\theta)$. | - |
| $g_t$ | Vector Gradient | Gradient Vector | Đạo hàm riêng của Loss đối với tham số tại bước $t$: $g_t = \nabla_{\theta} L(\theta_t)$. Chỉ ra hướng dốc đứng nhất. | - |
| $\eta$ (hoặc $\alpha$) | Tốc độ học | Learning Rate | Hệ số kiểm soát độ lớn của bước cập nhật tham số tại mỗi lần lặp. | $0.1$ (SGD), $0.001$ (Adam) |
| $v_t$ | Vận tốc (Moment bậc 1) | Velocity / 1st Moment | EMA của các gradient trong quá khứ, đại diện cho quán tính/hướng đi tích lũy. | - |
| $s_t$ (hoặc $v_t$) | Moment bậc 2 | 2nd Raw Moment | EMA của bình phương gradient ($g_t^2$), đại diện cho độ biến động để điều chỉnh lr thích ứng. | - |
| $\beta$ (hoặc $\beta_1$) | Hệ số phân rã moment 1 | 1st Moment Decay rate | Hệ số ma sát/kiểm soát quán tính của moment 1. Hệ số càng lớn, quán tính càng mạnh. | $0.9$ |
| $\beta_2$ | Hệ số phân rã moment 2 | 2nd Moment Decay rate | Hệ số suy giảm/kiểm soát bộ nhớ của moment 2. | $0.99$ (RMSprop), $0.999$ (Adam) |
| $\hat{m}_t$ | Moment 1 hiệu chỉnh | Bias-corrected 1st Moment | Kỳ vọng gradient không chệch sau khi khử lực kéo về 0 ở các bước lặp đầu tiên. | - |
| $\hat{v}_t$ | Moment 2 hiệu chỉnh | Bias-corrected 2nd Moment | Ước lượng phương sai gradient không chệch sau khi khử lực kéo về 0. | - |
| $\epsilon$ | Hệ số an toàn | Epsilon | Số cực nhỏ cộng vào mẫu số để tránh lỗi toán học chia cho 0. | $10^{-8}$ |
| $\lambda$ | Hệ số suy giảm trọng số | Weight Decay rate | Hệ số điều chuẩn trực tiếp lên trọng số để chống overfitting. | $0.01$ |
| $\odot$ | Phép nhân Hadamard | Hadamard Product | Phép nhân nhân từng phần tử tương ứng của hai ma trận hoặc vector có cùng kích thước. | - |

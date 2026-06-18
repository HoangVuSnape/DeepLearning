import json
import os

notebook_path = r"e:\DoCode\Master_Subject2026_TDTU\DeepLearning\notebook\optimizers_comparison.ipynb"

# 1. Thiết lập các cells của Notebook
cells = []

# Title & Intro
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# So sánh các bộ tối ưu hóa (Optimizers) trong MLP từ Scratch\n",
        "\n",
        "Notebook này trình bày chi tiết và hệ thống hóa hành trình phát triển của các thuật toán tối ưu hóa trong Học Sâu theo lộ trình 7 bước chuẩn mực học thuật:\n",
        "\n",
        "1. **Bài toán tối ưu & gradient**\n",
        "2. **Batch / SGD / Mini-batch**\n",
        "3. **Momentum & Nesterov**\n",
        "4. **AdaGrad & RMSProp**\n",
        "5. **Adam & AdamW**\n",
        "6. **Schedule · warmup · clipping**\n",
        "7. **Ví dụ số & so sánh lựa chọn**\n",
        "\n",
        "Toàn bộ mô hình mạng nơ-ron **MultiLayer Perceptron (MLP)** và các thuật toán tối ưu được hiện thực **từ đầu bằng NumPy** để phân loại tập dữ liệu phi tuyến **Half Moons**."
    ]
})

# Section 1: Bài toán tối ưu & gradient
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 1. Bài toán tối ưu hóa & Gradient trong Deep Learning\n",
        "\n",
        "Trong huấn luyện mạng nơ-ron, mục tiêu chính là giảm thiểu hàm chi phí/mất mát $L(\theta)$ bằng cách tìm bộ tham số tối ưu $\\theta^* = \\{W, b\\}$:\n",
        "$$\\theta^* = \\arg\\min_\\theta L(\\theta)$$\n",
        "\n",
        "**Gradient** (ký hiệu $\\nabla_\\theta L(\\theta)$) là một vector chứa tất cả các đạo hàm riêng của hàm mất mát theo từng tham số. Về mặt toán học, gradient chỉ ra **hướng dốc lên nhanh nhất** của hàm Loss tại điểm hiện tại. Do đó, để đi tìm cực tiểu (xuống đáy thung lũng), ta phải cập nhật tham số đi **ngược hướng gradient**:\n",
        "$$\\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta L(\\theta_t)$$\n",
        "Trong đó, $\\eta$ (Learning Rate) điều chỉnh kích thước bước đi."
    ]
})

# Section 2: Batch / SGD / Mini-batch
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Các biến thể Gradient Descent theo kích thước Batch\n",
        "\n",
        "Cách chúng ta chọn lượng dữ liệu để tính toán gradient tại mỗi bước lặp định hình nên 3 thuật toán phổ biến:\n",
        "\n",
        "1. **Batch Gradient Descent (BGD)**: Tính toán gradient trên toàn bộ $N$ mẫu dữ liệu huấn luyện.\n",
        "   - *Ưu điểm*: Hướng đi chuẩn xác, ít dao động.\n",
        "   - *Nhược điểm*: Cực kỳ tốn bộ nhớ và tính toán khi dữ liệu lớn ($N$ lớn).\n",
        "2. **Stochastic Gradient Descent (SGD)**: Lấy duy nhất $1$ mẫu dữ liệu ngẫu nhiên tại mỗi bước để tính gradient.\n",
        "   - *Ưu điểm*: Cực kỳ nhanh, tốn ít tài nguyên, có thể thoát khỏi cực tiểu cục bộ dễ hơn.\n",
        "   - *Nhược điểm*: Hướng đi rất \"nhiễu\" (zigzag dữ dội), khó hội tụ chính xác về đáy.\n",
        "3. **Mini-batch Gradient Descent**: Dung hòa hai phương pháp trên, chia tập dữ liệu thành các nhóm nhỏ kích thước $B$ ($1 < B < N$, thường chọn 32, 64, 128).\n",
        "   - Đây là sự lựa chọn tiêu chuẩn của mọi mạng học sâu hiện đại vì tối ưu hóa được năng lực xử lý song song của GPU/CPU và hướng đi hội tụ tương đối mượt mà."
    ]
})

# Section 3: Momentum & Nesterov
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Nhánh Quán tính: Momentum & Nesterov Accelerated Gradient (NAG)\n",
        "\n",
        "### 3.1 SGD với Momentum (Động lượng)\n",
        "SGD thường bị dao động mạnh ở các hướng dốc cao (high curvature). **Momentum** tích lũy các gradient trước đó đóng vai trò như quán tính của viên bi lăn, đẩy nhanh chuyển động ở hướng nhất quán và triệt tiêu dao động ở các hướng đổi dấu liên tục:\n",
        "$$v_t = \\beta_1 v_{t-1} + (1 - \\beta_1) g_t$$\n",
        "$$\\theta_{t+1} = \\theta_t - \\eta v_t$$\n",
        "Trong đó $\\beta_1 \\in [0, 1)$ là hệ số quán tính (thường chọn 0.9).\n",
        "\n",
        "### 3.2 Nesterov Accelerated Gradient (NAG)\n",
        "NAG cải tiến Momentum bằng cách tính toán gradient của hàm chi phí tại điểm dự đoán tiếp theo (look-ahead) thay vì tại điểm hiện hành. Giúp viên bi có khả năng \"phanh trước\" trước khi lao qua đỉnh cực tiểu:\n",
        "$$v_t = \\beta_1 v_{t-1} + g_t$$\n",
        "$$\\theta_{t+1} = \\theta_t - \\eta (g_t + \\beta_1 v_t)$$"
    ]
})

# Section 4: AdaGrad & RMSProp
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. Nhánh Tốc độ học thích ứng: AdaGrad & RMSProp\n",
        "\n",
        "Thay vì áp dụng cùng một learning rate cho mọi tham số, nhánh này thiết lập tốc độ học riêng cho từng trọng số dựa trên lịch sử cập nhật.\n",
        "\n",
        "### 4.1 AdaGrad (Adaptive Gradient)\n",
        "Chia learning rate cơ bản cho căn bậc hai của tổng bình phương gradient trong quá khứ:\n",
        "$$s_t = s_{t-1} + g_t^2$$\n",
        "$$\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{s_t} + \\epsilon} g_t$$\n",
        "- *Nhược điểm*: $s_t$ tăng đơn điệu liên tục $\\rightarrow$ learning rate bị giảm dần về 0 quá sớm, khiến mạng ngừng học trước khi hội tụ.\n",
        "\n",
        "### 4.2 RMSProp (Root Mean Squared Propagation)\n",
        "Khắc phục nhược điểm của AdaGrad bằng cách thay tổng tích lũy bằng trung bình trượt lũy thừa (EMA) của bình phương gradient, tập trung vào các bước đi gần nhất:\n",
        "$$s_t = \\beta_2 s_{t-1} + (1 - \\beta_2) g_t^2$$\n",
        "$$\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{s_t} + \\epsilon} g_t$$\n",
        "Trong đó $\\beta_2$ thường chọn là 0.99 hoặc 0.9."
    ]
})

# Section 5: Adam & AdamW
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Sự kết hợp hoàn hảo: Adam & Bản vá AdamW\n",
        "\n",
        "### 5.1 Adam (Adaptive Moment Estimation)\n",
        "Adam kết hợp cả động lượng (Moment bậc 1 $m_t$) và thích ứng learning rate (Moment bậc 2 $v_t$), đồng thời bổ sung hiệu chỉnh độ lệch (Bias Correction) vì $m_0, v_0$ khởi tạo bằng 0:\n",
        "$$m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t$$\n",
        "$$v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2$$\n",
        "$$\\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}$$\n",
        "$$\\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$$\n",
        "\n",
        "### 5.2 AdamW (Weight Decay Fix)\n",
        "Trong Adam truyền thống, việc trộn L2 regularization vào gradient gốc $g_t$ làm cho hệ số phạt này bị chia cho hệ số thích ứng $\\sqrt{\\hat{v}_t}$. Điều này làm sai lệch tác dụng của Weight Decay (phạt quá nhẹ các trọng số hoạt động mạnh và phạt quá nặng các trọng số ít hoạt động).\n",
        "\n",
        "**AdamW** sửa lỗi này bằng cách tách biệt hoàn toàn (decouple) Weight Decay khỏi bước điều chỉnh thích ứng:\n",
        "$$\\theta_{t+1} = \\theta_t - \\eta \\lambda \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t$$\n",
        "Trong đó $\\lambda$ là tham số Weight Decay."
    ]
})

# Section 6: Schedule · warmup · clipping
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6. Các kỹ thuật huấn luyện nâng cao\n",
        "\n",
        "### 6.1 Learning Rate Scheduler (Cosine Annealing)\n",
        "Thay vì giữ nguyên tốc độ học $\\eta$, ta cho $\\eta$ giảm dần theo hàm cosine về một giá trị tối thiểu để giúp mô hình hội tụ mượt mà và ổn định ở các epoch cuối.\n",
        "\n",
        "### 6.2 Learning Rate Warmup\n",
        "Trong các bước đầu huấn luyện, các moment của Adam chưa ổn định, việc dùng learning rate lớn dễ làm hỏng trọng số. Warmup giúp tăng dần learning rate tuyến tính từ 0 đến giá trị ban đầu trong một số bước lặp (steps) đầu tiên.\n",
        "\n",
        "### 6.3 Gradient Clipping\n",
        "Khi đạo hàm quá lớn (Gradient Explosion), bước nhảy tham số sẽ quá xa làm mô hình phân kỳ (nan loss). Gradient clipping giới hạn độ lớn (L2 norm) của gradient vector không vượt quá một ngưỡng $C$ cho trước:\n",
        "$$g_{\\text{clipped}} = g \\cdot \\min\\left(1, \\frac{C}{\\|g\\|_2}\\right)$$"
    ]
})

# Code Cell 1: Import libraries
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Bước 1: Import các thư viện cần thiết\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.datasets import make_moons\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "print(\"Đã import thành công các thư viện!\")"
    ]
})

# Code Cell 2: MLP class implementation
mlp_code = """# Bước 2: Định nghĩa mô hình MLP và thuật toán cập nhật của các Optimizers từ Scratch

class MLPClassifierWithOptimizers:
    \"\"\"Mô hình mạng MLP hỗ trợ nhiều bộ tối ưu hóa khác nhau và các kỹ thuật nâng cấp huấn luyện.\"\"\"
    
    def __init__(
        self,
        layer_sizes: list[int],
        learning_rate: float = 0.01,
        epochs: int = 1000,
        batch_size: int = 32,
        optimizer: str = "adam",
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        clip_norm: float = None,
        lr_scheduler: str = None,  # "cosine", "step", or None
        warmup_steps: int = 0,
    ):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer.lower()
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        
        # 1. Khởi tạo trọng số (Weights) và độ chệch (Biases) dùng He Initialization
        self.W = []
        self.b = []
        for i in range(1, len(layer_sizes)):
            limit = np.sqrt(2.0 / layer_sizes[i - 1])
            self.W.append(np.random.randn(layer_sizes[i], layer_sizes[i - 1]) * limit)
            self.b.append(np.zeros((layer_sizes[i], 1)))
            
        # 2. Khởi tạo trạng thái lịch sử của các bộ tối ưu hóa (velocity, adaptive states)
        self.v_W = [np.zeros_like(w) for w in self.W]
        self.v_b = [np.zeros_like(bias) for bias in self.b]
        self.m_W = [np.zeros_like(w) for w in self.W]
        self.m_b = [np.zeros_like(bias) for bias in self.b]
        self.s_W = [np.zeros_like(w) for w in self.W]
        self.s_b = [np.zeros_like(bias) for bias in self.b]
        
        self.loss_history = []
        self.accuracy_history = []
        
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        
    @staticmethod
    def _sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        return a * (1.0 - a)
        
    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        \"\"\"Lan truyền tiến qua các lớp.\"\"\"
        a_values = [X]
        z_values = []
        a = X
        for W_l, b_l in zip(self.W, self.b):
            z = W_l @ a + b_l
            a = self._sigmoid(z)
            z_values.append(z)
            a_values.append(a)
        return a_values, z_values
        
    def _backward(self, a_values: list[np.ndarray], y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        \"\"\"Lan truyền ngược sai số tính Gradient (dW, db) dựa trên Chain Rule.\"\"\"
        n_samples = y.shape[1]
        dW = []
        db = []
        
        # Tính lỗi ở lớp đầu ra
        a_out = a_values[-1]
        delta = (a_out - y) * self._sigmoid_derivative(a_out)
        
        num_layers = len(self.W)
        for l in reversed(range(num_layers)):
            a_prev = a_values[l]
            dW_l = (1.0 / n_samples) * (delta @ a_prev.T)
            db_l = (1.0 / n_samples) * np.sum(delta, axis=1, keepdims=True)
            
            # L2 Regularization (áp dụng trực tiếp vào gradient nếu không phải AdamW)
            if self.weight_decay > 0.0 and self.optimizer != "adamw":
                dW_l += self.weight_decay * self.W[l]
                
            dW.insert(0, dW_l)
            db.insert(0, db_l)
            
            if l > 0:
                delta = (self.W[l].T @ delta) * self._sigmoid_derivative(a_prev)
                
        return dW, db
        
    def _update_parameters(self, dW: list[np.ndarray], db: list[np.ndarray], t: int, lr: float) -> None:
        \"\"\"Cập nhật trọng số theo thuật toán tối ưu đã chọn.\"\"\"
        for l in range(len(self.W)):
            if self.optimizer == "sgd":
                self.W[l] -= lr * dW[l]
                self.b[l] -= lr * db[l]
                
            elif self.optimizer == "momentum":
                self.v_W[l] = self.beta1 * self.v_W[l] + (1 - self.beta1) * dW[l]
                self.v_b[l] = self.beta1 * self.v_b[l] + (1 - self.beta1) * db[l]
                self.W[l] -= lr * self.v_W[l]
                self.b[l] -= lr * self.v_b[l]
                
            elif self.optimizer == "nesterov":
                # PyTorch style Nesterov Accelerated Gradient
                self.v_W[l] = self.beta1 * self.v_W[l] + dW[l]
                self.v_b[l] = self.beta1 * self.v_b[l] + db[l]
                self.W[l] -= lr * (dW[l] + self.beta1 * self.v_W[l])
                self.b[l] -= lr * (db[l] + self.beta1 * self.v_b[l])
                
            elif self.optimizer == "adagrad":
                self.s_W[l] += dW[l] ** 2
                self.s_b[l] += db[l] ** 2
                self.W[l] -= lr * dW[l] / (np.sqrt(self.s_W[l]) + self.epsilon)
                self.b[l] -= lr * db[l] / (np.sqrt(self.s_b[l]) + self.epsilon)
                
            elif self.optimizer == "rmsprop":
                self.s_W[l] = self.beta2 * self.s_W[l] + (1 - self.beta2) * (dW[l] ** 2)
                self.s_b[l] = self.beta2 * self.s_b[l] + (1 - self.beta2) * (db[l] ** 2)
                self.W[l] -= lr * dW[l] / (np.sqrt(self.s_W[l]) + self.epsilon)
                self.b[l] -= lr * db[l] / (np.sqrt(self.s_b[l]) + self.epsilon)
                
            elif self.optimizer in ["adam", "adamw"]:
                self.m_W[l] = self.beta1 * self.m_W[l] + (1 - self.beta1) * dW[l]
                self.m_b[l] = self.beta1 * self.m_b[l] + (1 - self.beta1) * db[l]
                self.v_W[l] = self.beta2 * self.v_W[l] + (1 - self.beta2) * (dW[l] ** 2)
                self.v_b[l] = self.beta2 * self.v_b[l] + (1 - self.beta2) * (db[l] ** 2)
                
                # Bias correction
                m_W_hat = self.m_W[l] / (1.0 - self.beta1 ** t)
                m_b_hat = self.m_b[l] / (1.0 - self.beta1 ** t)
                v_W_hat = self.v_W[l] / (1.0 - self.beta2 ** t)
                v_b_hat = self.v_b[l] / (1.0 - self.beta2 ** t)
                
                # Decoupled Weight Decay của AdamW
                if self.optimizer == "adamw":
                    self.W[l] -= lr * self.weight_decay * self.W[l]
                    
                self.W[l] -= lr * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
                self.b[l] -= lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifierWithOptimizers":
        \"\"\"Huấn luyện mạng nơ-ron sử dụng Mini-batch GD và cấu hình tối ưu đã chọn.\"\"\"
        n_samples = X.shape[0]
        y_input = y.reshape(-1, 1) if len(y.shape) == 1 else y
        
        self.loss_history = []
        self.accuracy_history = []
        
        batches_per_epoch = int(np.ceil(n_samples / self.batch_size))
        total_steps = self.epochs * batches_per_epoch
        t = 0
        
        for epoch in range(1, self.epochs + 1):
            shuffled_indices = np.random.permutation(n_samples)
            X_shuffled = X[shuffled_indices]
            y_shuffled = y_input[shuffled_indices]
            
            epoch_loss = 0.0
            correct_preds = 0
            
            for i in range(0, n_samples, self.batch_size):
                t += 1
                
                # Slicing mini-batch
                X_batch = X_shuffled[i : i + self.batch_size].T
                y_batch = y_shuffled[i : i + self.batch_size].T
                
                # 1. Forward Pass
                a_values, _ = self._forward(X_batch)
                y_hat = a_values[-1]
                
                # Tính Loss (MSE)
                batch_loss = 0.5 * np.mean(np.sum((y_hat - y_batch) ** 2, axis=0))
                epoch_loss += batch_loss * (X_batch.shape[1] / n_samples)
                
                # Tính Accuracy
                preds = (y_hat >= 0.5).astype(int)
                correct_preds += np.sum(preds == y_batch)
                
                # 2. Backward Pass
                dW, db = self._backward(a_values, y_batch)
                
                # 3. Gradient Clipping
                if self.clip_norm is not None:
                    total_norm = np.sqrt(sum(np.sum(dw**2) for dw in dW) + sum(np.sum(dbb**2) for dbb in db))
                    if total_norm > self.clip_norm:
                        clip_coef = self.clip_norm / (total_norm + 1e-6)
                        for l in range(len(dW)):
                            dW[l] *= clip_coef
                            db[l] *= clip_coef
                
                # 4. Learning Rate Schedule & Warmup
                current_lr = self.lr
                if self.warmup_steps > 0 and t <= self.warmup_steps:
                    current_lr = self.lr * (t / self.warmup_steps)
                else:
                    if self.lr_scheduler == "cosine":
                        progress = (t - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
                        current_lr = self.lr * 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))
                    elif self.lr_scheduler == "step":
                        step_idx = (t - self.warmup_steps) / max(1, total_steps * 0.3)
                        current_lr = self.lr * (0.5 ** int(step_idx))
                        
                # 5. Parameter Update
                self._update_parameters(dW, db, t, current_lr)
                
            self.loss_history.append(epoch_loss)
            epoch_acc = correct_preds / n_samples
            self.accuracy_history.append(epoch_acc)
            
            # In kết quả định kỳ
            if epoch == 1 or epoch % 200 == 0 or epoch == self.epochs:
                print(f"Epoch {epoch:4d}/{self.epochs:4d} | Loss: {epoch_loss:.6f} | Train Acc: {epoch_acc:.2%}")
                
        return self
        
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        X_input = X.T
        a_values, _ = self._forward(X_input)
        probs = a_values[-1].T
        return (probs >= threshold).astype(int).ravel()

print("Đã định nghĩa thành công lớp MLP Classifier với đầy đủ Optimizers và nâng cấp!")"""

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [mlp_code]
})

# Section 7: Ví dụ số & so sánh lựa chọn - Intro
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 7. Thực nghiệm So sánh 7 bộ tối ưu hóa & Ví dụ tính số\n",
        "\n",
        "### 7.1 Bảng so sánh tổng hợp các thuật toán tối ưu hóa\n",
        "\n",
        "| Thuật toán | Cơ chế cốt lõi | Ưu điểm chính | Trường hợp ứng dụng tốt nhất |\n",
        "| :--- | :--- | :--- | :--- |\n",
        "| **SGD** | Đi ngược hướng gradient gốc | Rất đơn giản, ít tốn tính toán | Tinh chỉnh mô hình (Fine-tuning) |\n",
        "| **Momentum** | Tích lũy vector vận tốc lịch sử | Vượt điểm yên ngựa, giảm dao động | CNNs cổ điển (ví dụ ResNet) |\n",
        "| **Nesterov (NAG)**| Đo dốc tại điểm look-ahead | Hội tụ nhanh hơn Momentum thông thường | Mạng tích chập sâu, bài toán CV |\n",
        "| **AdaGrad** | Chia lr cho tổng bình phương gradient | Tự thích ứng lr từng tham số | Bài toán với dữ liệu thưa thớt (Sparse data) |\n",
        "| **RMSProp** | Chia lr cho EMA của bình phương gradient | Giải quyết lỗi triệt tiêu lr của AdaGrad | Mạng RNNs, LSTMs, học tăng cường |\n",
        "| **Adam** | Hợp nhất Momentum và RMSProp | Hội tụ cực nhanh, tự động hóa lr | Mặc định cho hầu hết mạng nơ-ron |\n",
        "| **AdamW** | Tách rời Weight Decay khỏi gradient | Khả năng tổng quát hóa trên tập test tốt | Huấn luyện **Transformers, LLMs, BERT, GPT** |"
    ]
})

# Code Cell 3: Data Preparation
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Bước 3: Tạo và phân tách tập dữ liệu\n",
        "np.random.seed(42)\n",
        "X, y = make_moons(n_samples=500, noise=0.2, random_state=42)\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "# Chuẩn hóa đặc trưng đầu vào\n",
        "scaler = StandardScaler()\n",
        "X_train = scaler.fit_transform(X_train)\n",
        "X_test = scaler.transform(X_test)\n",
        "\n",
        "print(f\"Kích thước tập Train: {X_train.shape}\")\n",
        "print(f\"Kích thước tập Test: {X_test.shape}\")"
    ]
})

# Code Cell 4: Training 7 Optimizers
train_compare_code = """# Bước 4: Thiết lập cấu hình và tiến hành huấn luyện so sánh 7 bộ tối ưu hóa

layer_sizes = [2, 10, 5, 1]
epochs = 1200
batch_size = 32

optimizers_config = {
    "SGD": {"optimizer": "sgd", "lr": 0.05, "weight_decay": 0.0},
    "Momentum": {"optimizer": "momentum", "lr": 0.05, "beta1": 0.9, "weight_decay": 0.0},
    "Nesterov": {"optimizer": "nesterov", "lr": 0.05, "beta1": 0.9, "weight_decay": 0.0},
    "AdaGrad": {"optimizer": "adagrad", "lr": 0.01, "weight_decay": 0.0},
    "RMSprop": {"optimizer": "rmsprop", "lr": 0.005, "beta2": 0.99, "weight_decay": 0.0},
    "Adam": {"optimizer": "adam", "lr": 0.005, "beta1": 0.9, "beta2": 0.999, "weight_decay": 1e-4},
    "AdamW": {"optimizer": "adamw", "lr": 0.005, "beta1": 0.9, "beta2": 0.999, "weight_decay": 1e-2},
}

results = {}

print("=" * 65)
print(" BẮT ĐẦU HUẤN LUYỆN SO SÁNH 7 OPTIMIZERS VỚI MINI-BATCH (SIZE = 32)")
print("=" * 65)

for name, config in optimizers_config.items():
    print(f"\\n>>> Đang chạy bộ tối ưu: {name}...")
    np.random.seed(42)  # Đồng bộ khởi tạo trọng số để công bằng
    mlp = MLPClassifierWithOptimizers(
        layer_sizes=layer_sizes,
        learning_rate=config["lr"],
        epochs=epochs,
        batch_size=batch_size,
        optimizer=config["optimizer"],
        beta1=config.get("beta1", 0.9),
        beta2=config.get("beta2", 0.999),
        weight_decay=config.get("weight_decay", 0.0),
        clip_norm=1.0,  # Bổ sung Gradient Clipping hỗ trợ hội tụ
        lr_scheduler="cosine",  # Bổ sung Cosine lr scheduler
        warmup_steps=50,  # Bổ sung 50 warmup steps
    )
    mlp.fit(X_train, y_train)
    
    # Đánh giá trên tập test
    test_preds = mlp.predict(X_test)
    test_acc = float(np.mean(test_preds == y_test))
    print(f"--> KẾT QUẢ {name}: Test Accuracy = {test_acc:.2%} | Final Loss = {mlp.loss_history[-1]:.6f}")
    
    results[name] = {
        "loss_history": mlp.loss_history,
        "accuracy_history": mlp.accuracy_history,
        "test_accuracy": test_acc,
    }"""

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [train_compare_code]
})

# Code Cell 5: Visualization
vis_code = """# Bước 5: Trực quan hóa đồ thị so sánh Loss và Accuracy của cả 7 bộ tối ưu
plt.figure(figsize=(15, 6))

colors = {
    "SGD": "blue",
    "Momentum": "orange",
    "Nesterov": "purple",
    "AdaGrad": "brown",
    "RMSprop": "cyan",
    "Adam": "green",
    "AdamW": "red"
}

# Đồ thị Loss
plt.subplot(1, 2, 1)
for name, data in results.items():
    plt.plot(data["loss_history"], label=f"{name} (Test Acc: {data['test_accuracy']:.2%})", color=colors.get(name, "black"), linewidth=2.0)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("So sánh Lịch sử Loss (Mini-batch & Cosine Scheduler)")
plt.grid(True, alpha=0.3)
plt.legend()

# Đồ thị Accuracy
plt.subplot(1, 2, 2)
for name, data in results.items():
    plt.plot(data["accuracy_history"], label=name, color=colors.get(name, "black"), linewidth=2.0)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("So sánh Lịch sử Accuracy (Huấn luyện)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("optimizers_loss_comparison.png", dpi=150)
plt.show()"""

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [vis_code]
})

# Section 7.2: Numerical example
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 7.2 Ví dụ Tính Toán Số Học Từng Bước của Adam (t = 1)\n",
        "\n",
        "Để kiểm chứng thuật toán hoạt động chính xác bên dưới mã nguồn, chúng ta chạy một bài toán 2 chiều đơn giản với các số liệu thực tế tại bước lặp đầu tiên ($t = 1$)."
    ]
})

# Code Cell 6: Numerical Verification
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Thiết lập thông số theo ví dụ tính toán thủ công\n",
        "theta = np.array([1.5, -2.0])\n",
        "g1 = np.array([0.5, -1.2])\n",
        "\n",
        "# Siêu tham số\n",
        "beta1 = 0.9\n",
        "beta2 = 0.999\n",
        "eta = 0.1\n",
        "epsilon = 1e-8\n",
        "\n",
        "# Trạng thái ban đầu t = 0\n",
        "m0 = np.array([0.0, 0.0])\n",
        "v0 = np.array([0.0, 0.0])\n",
        "\n",
        "print(\"=== BƯỚC TÍNH SỐ THỰC TẾ CHO ADAM (t = 1) ===\")\n",
        "print(f\"Vector tham số khởi đầu theta_1: {theta}\")\n",
        "print(f\"Vector Gradient tại bước 1 g_1:  {g1}\")\n",
        "\n",
        "# Bước 1: Tính Moment bậc 1 (m1)\n",
        "m1 = beta1 * m0 + (1.0 - beta1) * g1\n",
        "print(f\"\\n1. Moment bậc 1 (chưa hiệu chỉnh) m_1:\\n   m_1 = {m1}\")\n",
        "\n",
        "# Bước 2: Tính Moment bậc 2 (v1)\n",
        "v1 = beta2 * v0 + (1.0 - beta2) * (g1 ** 2)\n",
        "print(f\"2. Moment bậc 2 (chưa hiệu chỉnh) v_1:\\n   v_1 = {v1}\")\n",
        "\n",
        "# Bước 3: Hiệu chỉnh độ lệch (Bias Correction) với t = 1\n",
        "t = 1\n",
        "m1_hat = m1 / (1.0 - beta1 ** t)\n",
        "v1_hat = v1 / (1.0 - beta2 ** t)\n",
        "print(f\"3. Moment hiệu chỉnh độ lệch tại t = 1:\\n   m1_hat = {m1_hat} (Khớp với g_1)\\n   v1_hat = {v1_hat} (Khớp với g_1^2)\")\n",
        "\n",
        "# Bước 4: Hướng cập nhật chuẩn hóa thang đo (U1)\n",
        "U1 = m1_hat / (np.sqrt(v1_hat) + epsilon)\n",
        "print(f\"4. Hướng cập nhật U_1:\\n   U_1 = {U1} (Bằng dấu của gradient sign(g_1))\")\n",
        "\n",
        "# Bước 5: Cập nhật tham số thu được theta_2\n",
        "theta_2 = theta - eta * U1\n",
        "print(f\"5. Vector tham số mới theta_2 sau cập nhật:\\n   theta_2 = {theta_2} (Kỳ vọng: [1.4, -1.9])\")\n",
        "\n",
        "assert np.allclose(theta_2, np.array([1.4, -1.9])), \"Lỗi: Kết quả tính toán không khớp!\""
    ]
})

# Construct notebook structure
notebook_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.7.3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# Ensure directory exists
os.makedirs(os.path.dirname(notebook_path), exist_ok=True)

# Write notebook file
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=1, ensure_ascii=False)

print(f"Đã cập nhật Notebook thành công tại: {notebook_path}")

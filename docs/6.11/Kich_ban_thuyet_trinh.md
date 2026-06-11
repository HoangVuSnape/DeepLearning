# 🎤 Kịch bản thuyết trình — Optimizers trong Deep Learning

**Tổng thời lượng dự kiến:** ~25–28 phút (trung bình ~1.2–1.5 phút/slide)
**Đối tượng:** Sinh viên/đồng nghiệp đã biết cơ bản về mạng nơ-ron
**Mục tiêu:** Hiểu lý do tại sao chúng ta dùng Adam/AdamW thay vì SGD thuần

👉 Tra cứu nhanh các ký hiệu toán học tại: [Cheat Sheet: Ký hiệu & Ý nghĩa toán học trong Tối ưu hóa](cheatsheet.md)

---


## 🎯 Mẹo thuyết trình chung

- **Đừng đọc công thức** — chỉ chỉ tay vào, nói ý nghĩa. Ai muốn xem chi tiết tự đọc trên slide.
- Khi sang một slide mới, **dừng 1 giây** để người nghe nhìn đã rồi mới nói.
- Mỗi khi xong một ý lớn, có thể hỏi: *"Tới đây có ai muốn hỏi gì không?"* — nhưng cuối hẵng hỏi để tránh đứt mạch.
- Câu nối chuyển slide quan trọng hơn nội dung trong slide. Tập kỹ câu nối.

---

# Phần 1 — Đặt vấn đề (Slide 1–6)

## 📍 Slide 1 — Trang bìa  *(~45 giây)*

> "Chào mọi người. Hôm nay mình sẽ chia sẻ về một thành phần mà *gần như mọi mô hình deep learning hiện đại đều cần* — nhưng lại ít khi được nói đến: **Optimizer**.
>
> Khi train một mạng nơ-ron, có ba thứ chính: **dữ liệu**, **kiến trúc mạng**, và **cách tối ưu hóa**. Hai cái đầu được nói rất nhiều — bài báo nào về Transformer, ResNet, đều nhấn vào kiến trúc. Nhưng cái thứ ba — *Optimizer* — mới là thứ quyết định mô hình của bạn có hội tụ hay không.
>
> Mục tiêu của buổi hôm nay là đi qua **hành trình tiến hóa** của các Optimizer: từ SGD đơn giản nhất, đến AdamW — thứ mà GPT, BERT, ViT đều đang dùng. Mình sẽ giải thích **tại sao** từng cải tiến ra đời, chứ không chỉ liệt kê công thức."

**🔗 Chuyển:** *"Trước khi vào Optimizer, mình muốn cả lớp nhìn lại một bức tranh tổng thể đã."*

---

## 📍 Slide 2 — Vòng lặp huấn luyện  *(~60 giây)*

> "Đây là **vòng lặp** của mọi quá trình huấn luyện deep learning. Có 4 mắt xích:
>
> 1. **Dữ liệu và mô hình** — đầu vào.
> 2. **Hàm mất mát (Loss function)** — đo mô hình hiện tại sai bao nhiêu.
> 3. **Optimizer** — *cỗ máy tính toán* — quyết định: bước tiếp theo đi về đâu, và đi mạnh cỡ nào.
> 4. **Cập nhật trọng số** — và quay lại bước 1.
>
> Câu ở dưới mình muốn nhấn mạnh: ***Optimizer không tạo ra kiến trúc mạng, nhưng nó quyết định tốc độ và khả năng sống sót của mạng trên hành trình tìm kiếm điểm hoàn hảo.*** Một kiến trúc tốt mà chọn sai optimizer thì có thể không bao giờ hội tụ. Ngược lại, một kiến trúc trung bình với optimizer tốt vẫn ra được kết quả khá."

**🔗 Chuyển:** *"Vậy cái 'cỗ máy' đó hoạt động ra sao? Bắt đầu từ ý tưởng đơn giản nhất."*

---

## 📍 Slide 3 — Gradient Descent: Trực giác hình học  *(~75 giây)*

> "Tưởng tượng hàm Loss là một thung lũng — trục ngang là *trọng số* (Weight), trục đứng là *chi phí* (Cost). Mục tiêu của ta là tìm cái đáy thung lũng — chỗ Cost nhỏ nhất.
>
> Bắt đầu, ta đứng ở một điểm bất kỳ — *Initial Weight*. Tại điểm này, mình tính **độ dốc của hàm chi phí** — chính là **gradient**, đường đứt nét đen. Gradient cho biết *hướng đi lên dốc* — vậy nếu mình muốn đi xuống đáy, mình phải đi **ngược lại** hướng đó.
>
> Mỗi lần đi một bước nhỏ — *Incremental Step* — rồi tính lại gradient ở điểm mới, đi tiếp. Cứ thế cho đến khi gradient gần bằng 0 — nghĩa là mình đã ở đáy thung lũng, *Minimum Cost*.
>
> Đây là **ý tưởng cốt lõi** của Gradient Descent. Toàn bộ slide sau này chỉ là biến thể tinh vi hơn của ý tưởng này thôi."

**🔗 Chuyển:** *"Nhưng để có gradient mà đi, ta phải tính nó cho từng lớp trong mạng. Và đó là vai trò của Backpropagation."*

---

## 📍 Slide 4 — Toán học Backpropagation  *(~90 giây)*

> "Đây là phần *toán học* — nhưng đừng sợ, mình giải thích bằng *trực giác*.
>
> Backpropagation trả lời câu hỏi: ***'Lớp ẩn ở giữa mạng đã sai bao nhiêu?'*** Bởi vì ta chỉ biết sai số ở lớp đầu ra (so với label), còn các lớp ẩn thì không có label để so. Phải *suy ngược về*.
>
> **Bước 1** — định nghĩa: δˡ là *đạo hàm của Loss theo đầu vào tuyến tính zˡ của lớp l*. Đây là 'thông điệp sai số' của lớp l.
>
> **Bước 2** — qua activation: vì sau zˡ ta cho qua hàm kích hoạt g (như ReLU), nên chain rule cho ta nhân Hadamard với đạo hàm g'(zˡ).
>
> **Bước 3** — ở đây là *cái đẹp của thuật toán*: ∂L/∂aˡ = (Wˡ⁺¹)ᵀ δˡ⁺¹. Nghĩa là **sai số của lớp sau, nhân ngược với ma trận trọng số lớp sau, sẽ cho ra mức độ trách nhiệm của từng nơ-ron lớp này**.
>
> **Bước 4** — ghép Bước 2 và 3 lại, ta có công thức tổng quát màu đỏ ở góc dưới. *Sai số tầng ẩn = sai số lớp sau ⊗ trọng số ⊗ đạo hàm activation*. Cứ thế lan truyền ngược từ output về input.
>
> Điểm cần nhớ: Backprop là cách **tái sử dụng kết quả** — không phải tính từng đạo hàm riêng lẻ, mà dùng chain rule để tính một lần, truyền ngược."

**🔗 Chuyển:** *"Vậy đến đây, có ba khái niệm dễ bị lẫn lộn. Mình tách rõ ra ở slide tiếp theo."*

---

## 📍 Slide 5 — Gradient vs Backpropagation vs Gradient Descent  *(~75 giây)*

> "Ba khái niệm này nhiều người dùng lẫn lộn. Mình tách rõ:
>
> - **Gradient** là một *đại lượng toán học* — một **vector** chứa đạo hàm riêng theo từng tham số. Nó trả lời câu hỏi *'Tham số W, b cần thay đổi bao nhiêu để giảm Loss?'*
>
> - **Backpropagation** là một *giải thuật máy tính* — nó **không phải gradient**, mà là **cách tính** gradient nhanh và tiết kiệm bộ nhớ.
>
> - **Gradient Descent** là một *phương pháp tối ưu* — sau khi đã có gradient rồi, ta dùng nó như thế nào để cập nhật trọng số.
>
> Ba khái niệm này — *là gì, tính ra sao, dùng ra sao* — ba câu hỏi khác nhau, ba trả lời khác nhau.
>
> Sơ đồ ở dưới tóm tắt toàn bộ vòng lặp: **Forward Pass** (tính Loss) → **Backpropagation** (chain rule) → **Gradient** (∇L) → **Optimizer** (SGD, Adam...) → **Cập nhật θ** — rồi lại quay về Forward.
>
> Phần còn lại của buổi hôm nay, chúng ta sẽ tập trung vào hộp **Optimizer** ở giữa sơ đồ này."

**🔗 Chuyển:** *"Bây giờ, vào câu chuyện chính: optimizer đã tiến hóa như thế nào theo thời gian."*

---

## 📍 Slide 6 — Bản đồ tiến hóa Optimizers  *(~75 giây)*

> "Đây là **cây phả hệ** của các optimizer hiện đại. Mọi thứ đều xuất phát từ Gradient Descent gốc, rồi tiến hóa theo **hai nhánh** song song.
>
> **Nhánh xanh — Quán tính (Momentum):** giải quyết bài toán *zigzag* và *kẹt ở local minima*. Ý tưởng: cho viên bi có động lượng.
>
> **Nhánh đỏ — Thích ứng (Adaptive):** giải quyết bài toán *learning rate cố định không hợp lý*. Ý tưởng: mỗi tham số có learning rate riêng, tự điều chỉnh.
>
> Hai nhánh chạy song song một thời gian — cho đến năm 2014, **Diederik Kingma và Jimmy Ba** ghép hai nhánh lại thành **Adam** — viết tắt của *Adaptive Moment Estimation*. Từ đó Adam trở thành 'optimizer quốc dân'.
>
> Năm 2017, Loshchilov và Hutter sửa một lỗi nhỏ trong cách Adam xử lý weight decay, ra **AdamW** — và đây là thứ mà mọi LLM hiện đại đang dùng.
>
> Bây giờ mình sẽ đi từng nhánh — *bắt đầu từ gốc*."

**🔗 Chuyển:** *"Quay lại điểm xuất phát — SGD."*

---

# Phần 2 — Nhánh Quán tính (Slide 7–11)

## 📍 Slide 7 — SGD: Gốc rễ & Điểm mù  *(~75 giây)*

> "SGD — *Stochastic Gradient Descent*. 'Stochastic' nghĩa là *ngẫu nhiên* — vì thay vì tính gradient trên toàn bộ tập dữ liệu (rất nặng), ta tính trên một mini-batch nhỏ.
>
> Công thức cực kỳ đơn giản: ***θ mới = θ cũ trừ đi learning rate nhân gradient.*** Một dòng code thôi.
>
> Nhưng SGD có **điểm mù**: trong vùng hàm Loss có 'thung lũng hẹp' — nghĩa là dốc theo một hướng nhưng phẳng theo hướng khác — SGD sẽ **dao động qua lại** hai bên sườn dốc thay vì đi nhanh dọc theo đáy. Hình bên phải minh họa cái zigzag đó.
>
> Đây là **động lực ra đời** của tất cả optimizer sau này: làm sao đi mượt hơn trong thung lũng hẹp?"

**🔗 Chuyển:** *"Cùng xem công thức chi tiết của SGD."*

---

## 📍 Slide 8 — SGD: Công thức  *(~45 giây)*

> "Hai bước:
>
> **Bước 1**: Tính gradient g_t — đạo hàm của Loss theo θ tại bước t. Backprop làm việc này.
>
> **Bước 2**: Cập nhật θ — đi ngược hướng gradient một khoảng η. η ở đây là *learning rate* — siêu tham số quan trọng nhất.
>
> Không có gì để nhớ hơn ngoài hai dòng này. SGD là **viên gạch nền** của mọi optimizer sau này — *các thuật toán mới chỉ thay đổi cách tính 'cái g_t' đó thôi*."

**🔗 Chuyển:** *"Và cải tiến đầu tiên là cho viên bi của ta có quán tính."*

---

## 📍 Slide 9 — Momentum: Cú hích từ Quán tính  *(~90 giây)*

> "Hãy tưởng tượng vật lý: nếu bạn thả một viên bi từ trên dốc xuống, **nó không dừng ngay khi gặp gồ ghề** — nó có quán tính, nó vượt qua được.
>
> SGD thuần thì như một *vật cứng đặt tại chỗ*: gradient bằng 0 là dừng ngay. Còn SGD với Momentum thì như *viên bi có khối lượng*: vẫn lăn được qua các đoạn gradient nhỏ, vẫn vượt được local minimum.
>
> Nhìn hình giữa — *Ẩn dụ vật lý*: viên bi A có quán tính, vượt qua đỉnh dốc E để lăn về đáy thật C. Viên bi B không có quán tính, dừng ngay tại điểm yên ngựa D.
>
> Bên phải là **kết quả thực nghiệm**: cùng một hàm Loss, SGD không Momentum kẹt ở local min phía bên phải. SGD có Momentum vượt qua được, về cực tiểu toàn cục.
>
> Đây là **ý tưởng đơn giản mà hiệu quả** — chỉ cần thêm một biến trạng thái."

**🔗 Chuyển:** *"Công thức cũng chỉ thêm một dòng so với SGD."*

---

## 📍 Slide 10 — Momentum: Công thức  *(~60 giây)*

> "Hai bước thôi:
>
> **Bước 1**: Tính **vận tốc v_t**. Đây là 'trung bình trượt lũy thừa' — *Exponential Moving Average* — của gradient.
>
> Diễn giải: β khoảng 0.9 — nghĩa là vận tốc mới = 90% vận tốc cũ + 10% gradient hiện tại. Quá khứ chiếm chủ đạo, hiện tại chỉ điều chỉnh nhẹ. **Đây chính là 'quán tính'** dạng số học.
>
> **Bước 2**: Cập nhật θ — nhưng thay vì dùng g_t, ta dùng v_t. Vậy thôi.
>
> Hệ quả ở dưới — *ý quan trọng nhất bạn cần nhớ*: hướng nào *nhất quán* (gradient luôn cùng dấu) → v_t cộng dồn → tăng tốc. Hướng nào *dao động* (gradient đổi dấu liên tục) → v_t triệt tiêu lẫn nhau → giảm dao động. **Vậy là vừa nhanh hơn vừa mượt hơn SGD thuần.**"

**🔗 Chuyển:** *"Có một minh họa rất thuyết phục cho điều này — mình bật lên cho mọi người xem."*

---

## 📍 Slide 11 — Tác động Momentum: vượt cực tiểu cục bộ  *(~60 giây — chờ animation chạy)*

> "*[chỉ tay vào hai animation]*
>
> Hai animation này chạy trên cùng một hàm Loss: f(x) = x² + 10·sin(x) — có nhiều cực tiểu cục bộ.
>
> **Bên trái** — SGD thuần, không Momentum: viên bi đỏ trượt xuống cái hố gần nhất rồi *kẹt ở đó*. Không thoát được. Đây là **vấn đề local minima** — nỗi sợ kinh điển của Deep Learning.
>
> **Bên phải** — SGD với Momentum, hệ số γ = 0.9: viên bi đỏ lăn qua cái hố thứ nhất, nhờ quán tính nó vượt được đỉnh dốc, và **rơi xuống cực tiểu toàn cục** ở giữa.
>
> *[dừng 2 giây]*
>
> Cùng một bài toán. Cùng một learning rate. Chỉ thêm một dòng v_t. Kết quả khác hẳn. Đây là **sức mạnh của một ý tưởng đơn giản**."

**🔗 Chuyển:** *"Đó là nhánh quán tính. Bây giờ qua nhánh thứ hai — học thích ứng."*

---

# Phần 3 — Nhánh Thích ứng (Slide 12–13)

## 📍 Slide 12 — RMSprop: Sự tinh tế của Bản đồ  *(~75 giây)*

> "Nhánh thứ hai giải quyết một vấn đề khác: **mỗi tham số có một 'tầm quan trọng' khác nhau** — nhưng SGD và Momentum dùng **chung một learning rate η** cho toàn bộ mạng. Hợp lý không? Không hẳn.
>
> Người ta nghĩ ra **AdaGrad** trước — *Adaptive Gradient* — cho mỗi tham số có learning rate riêng. Nhưng AdaGrad có lỗi chí mạng: nó **tích lũy tổng bình phương gradient từ ngày đầu tiên** — nên càng về sau, learning rate càng tiến về 0, và mô hình **dừng học giữa chừng**.
>
> **Geoffrey Hinton** đề xuất RMSprop để sửa lỗi này — *bằng cách dùng đúng kỹ thuật Momentum đã dùng*: trung bình trượt lũy thừa thay vì tổng tích lũy. Chỉ quan tâm đến quá khứ *gần*.
>
> Nhìn công thức ở giữa: learning rate hiệu dụng = η chia cho căn của s_t. Hệ quả ở phải: tham số nào dao động mạnh → s_t lớn → tự động **phanh** (giảm tốc độ học). Tham số nào dao động yếu → s_t nhỏ → **tăng tốc**. Như một bản đồ địa hình — đoạn dốc thì đi chậm, đoạn phẳng thì đi nhanh."

**🔗 Chuyển:** *"Công thức chi tiết."*

---

## 📍 Slide 13 — RMSprop: Công thức  *(~50 giây)*

> "Để hiện thực hóa cái 'cảm biến dao động' này, RMSprop chỉ cần đúng 2 bước tính toán đơn giản:
>
> **Bước 1**: Chúng ta tính $s_t$ – đây là trung bình trượt lũy thừa của *bình phương gradient*. Các bạn lưu ý chữ 'bình phương' nhé. Tại sao lại bình phương? Vì ở đây ta chỉ quan tâm đến **độ lớn** của dao động dốc gắt hay bằng phẳng, chứ không quan tâm nó đang dốc lên hay dốc xuống.
>
> **Bước 2**: Khi cập nhật trọng số, ta lấy learning rate cơ bản chia cho căn bậc hai của $s_t$. Số $\epsilon$ ở đây rất nhỏ, chỉ đóng vai trò bảo vệ công thức khỏi bị lỗi chia cho 0.
>
> Nhìn lại một chút để thấy bức tranh toàn cảnh: *Momentum* thì cộng dồn gradient để tìm **hướng đi**, còn *RMSprop* thì cộng dồn bình phương gradient để vẽ **bản đồ địa hình**. Một bên cho ta lực đẩy, một bên cho ta chân phanh.
>
> *[Dừng khoảng 2 giây để tạo kịch tính]*
>
> Vậy thì... có một câu hỏi hiển nhiên đặt ra: Tại sao chúng ta không kết hợp cả hai? Vừa có hướng đi tốt của Momentum, vừa có chân phanh thông minh của RMSprop?"

**🔗 Chuyển:** *"Và đúng là người ta đã làm điều đó."*

---

# Phần 4 — Adam và hậu duệ (Slide 14–17)

## 📍 Slide 14 — Sự Hợp nhất Tối thượng: Adam  *(~75 giây)*

> "Năm 2014, Adam ra đời từ ý tưởng đơn giản: ***lấy cái tốt nhất của cả hai nhánh***.
>
> - Từ **Momentum**: lưu trung bình trượt của *gradient* — gọi là **Moment bậc 1** — cho biết *hướng đi*.
>
> - Từ **RMSprop**: lưu trung bình trượt của *bình phương gradient* — gọi là **Moment bậc 2** — cho biết *độ lớn* để điều chỉnh learning rate.
>
> Hai cái này không xung đột nhau — một bên là vector hướng, một bên là vector scale. Ghép vào là được.
>
> Ẩn dụ ở dưới mình rất thích: *Momentum như một quả cầu lăn tự do — dễ bị trượt quá đích. Adam như một quả cầu nặng có ma sát — vẫn có quán tính, nhưng dừng được đúng chỗ.*
>
> Vì sao thêm 'ma sát'? Vì có yếu tố adaptive learning rate — gradient lớn thì tự phanh."

**🔗 Chuyển:** *"Cùng mổ xẻ Adam — 5 bước."*

---

## 📍 Slide 15 — Giải phẫu Adam  *(~120 giây — slide quan trọng nhất)*

> "Đây là slide **quan trọng nhất** của buổi hôm nay. Mình đi chậm.
>
> **Bước 1 — Tính gradient.** Giống y SGD, backprop làm việc này.
>
> **Bước 2 — Moment bậc 1, m_t.** Đây *đúng là* công thức Momentum. β₁ thường = 0.9.
>
> **Bước 3 — Moment bậc 2, v_t.** Đây *đúng là* công thức RMSprop, dùng bình phương gradient. β₂ thường = 0.999 — gần 1 hơn nhiều, vì ta muốn lấy trung bình dài hạn hơn.
>
> **Bước 4 — Hiệu chỉnh độ lệch (Bias Correction).** Đây là chi tiết tinh tế.
>
> Vấn đề: ở bước đầu tiên, m_0 và v_0 được khởi tạo bằng 0. Vì β₁ và β₂ gần 1, nên trong vài bước đầu, m_t và v_t bị **kéo lệch về 0** — *không phản ánh đúng* trung bình thực tế.
>
> Giải pháp: chia cho (1 − β^t). Khi t nhỏ, (1 − β^t) cũng nhỏ → mẫu nhỏ → bù lại được. Khi t lớn, β^t → 0 → mẫu → 1 → hiệu chỉnh tự động tắt. **Một mẹo toán học rất duyên dáng.**
>
> **Bước 5 — Cập nhật**. m̂_t cho hướng, √v̂_t cho scale. Vậy là xong.
>
> Tổng kết Adam trong một câu: ***'Đi theo hướng trung bình của gradient gần đây, với tốc độ tự điều chỉnh dựa trên độ ồn của gradient'.***"

**🔗 Chuyển:** *"Vì sao công thức này thành công đến mức trở thành mặc định của cả ngành?"*

---

## 📍 Slide 16 — Tứ Trụ Sức Mạnh của Adam  *(~75 giây)*

> "Adam thành công nhờ 4 tính chất mà ít optimizer khác có đủ:
>
> **1. Tự thích ứng.** Bạn không cần tinh chỉnh learning rate tỉ mỉ. Cứ để mặc định 0.001 — đa số trường hợp đều chạy được.
>
> **2. Kháng nhiễu tốt.** Khi gradient nhiều noise (batch nhỏ, dữ liệu lệch), Adam vẫn ổn định nhờ trung bình hóa qua m và v.
>
> **3. Hiệu năng cao.** Chi phí thêm chỉ là *lưu hai ma trận m và v cùng kích thước với θ*. Bộ nhớ x2 — nhưng tính toán cực rẻ.
>
> **4. Vượt điểm yên ngựa.** Saddle point là vùng gradient gần 0 nhưng không phải minimum — chỗ mà SGD hay kẹt. Adam thoát được nhờ momentum.
>
> *[dừng]*
>
> Vì 4 lý do này, Adam trở thành **mặc định mới** của cộng đồng. PyTorch, TensorFlow, JAX — mở documentation ra là thấy Adam ở đầu danh sách."

**🔗 Chuyển:** *"Nhưng — và đây là chữ 'nhưng' quan trọng — Adam có một lỗi nhỏ mà mãi 3 năm sau mới có người sửa."*

---

## 📍 Slide 17 — AdamW: Bản Vá Hoàn Hảo  *(~100 giây)*

> "Đầu tiên, chúng ta cùng nói về một nghịch lý từng khiến cả cộng đồng Deep Learning đau đầu:
> Khoảng năm 2016 - 2017, khi huấn luyện các mô hình Computer Vision lớn (như ResNet), các kỹ sư phát hiện một điều rất lạ: **Mặc dù Adam giúp mô hình hội tụ nhanh hơn trên tập Train, nhưng khi đánh giá trên tập Test, SGD kết hợp Momentum lại cho độ chính xác cao hơn và ít bị overfitting hơn.**
> Tại sao một cỗ máy tối tân như Adam lại đi giật lùi so với SGD truyền thống?
>
> Câu trả lời nằm ở thủ phạm mang tên: **L2 Regularization** – kỹ thuật phạt trọng số để chống overfitting.
> 
> Trong các thư viện như PyTorch hay TensorFlow thời kỳ đầu, người ta thực hiện L2 bằng cách cộng thêm bình phương trọng số vào hàm Loss. Khi đạo hàm, phần phạt này sẽ được gộp chung (trộn lẫn) vào với gradient gốc $g_t$.
>
> Điểm mấu chốt nằm ở đây: Trong Adam, hướng cập nhật sẽ bị chia cho căn bậc hai của lịch sử gradient ($\sqrt{v_t}$). Khi ta trộn L2 vào gradient, **lực phạt L2 này cũng bị chia cho $\sqrt{v_t}$**.
> Hậu quả là: 
> * Trọng số nào hoạt động rất mạnh (gradient lịch sử lớn) $\rightarrow$ bị chia cho số lớn $\rightarrow$ hầu như **không bị phạt**.
> * Trọng số nào ít hoạt động (gradient lịch sử nhỏ) $\rightarrow$ bị chia cho số nhỏ $\rightarrow$ lại bị **phạt rất nặng**.
> Sự méo mó toán học này làm mất đi hoàn toàn ý nghĩa ban đầu của L2 Regularization.
>
> Đến năm 2017, thuật toán **AdamW** ra đời để sửa chữa lỗi này bằng một ý tưởng cực kỳ đơn giản: **Tách biệt hoàn toàn (Decouple) phần suy giảm trọng số khỏi công thức gradient.** 
> Ta vẫn chuẩn hóa gradient gốc như bình thường, còn phần phạt weight decay sẽ được trừ trực tiếp vào trọng số ở bước cuối cùng, không thông qua bộ chia của Adam nữa.
>
> Sự thay đổi nhỏ này đã mang lại kết quả kỳ diệu: **AdamW giữ nguyên tốc độ hội tụ siêu nhanh của Adam, đồng thời đạt được khả năng khái quát hóa xuất sắc của SGD.**
> Đây chính là lý do vì sao ngày nay, 100% các mô hình ngôn ngữ lớn như GPT, BERT, LLaMA hay Claude đều bắt buộc phải dùng AdamW chứ không dùng Adam truyền thống."

**🔗 Chuyển:** *"Lý thuyết đến đây là đủ. Mình có một thí nghiệm thực tế để chứng minh."*

---

# Phần 5 — Thực nghiệm & Kết luận (Slide 18–20)

## 📍 Slide 18 — Bằng chứng Thực nghiệm  *(~75 giây)*

> "Đây là kết quả thí nghiệm mình tự code bằng NumPy — *không dùng framework nào*. Một mạng MLP nhỏ, kiến trúc [2, 10, 5, 1], train trên tập **Half Moons** — bài toán phân loại phi tuyến cổ điển, trong 1500 epochs.
>
> Bốn optimizer chạy song song trên *cùng dữ liệu, cùng kiến trúc, cùng learning rate gốc*.
>
> Bên trái — **biểu đồ Loss**: Adam (đường đỏ) và RMSprop (xanh lá) giảm Loss **nhanh hơn hẳn** SGD trong 200 epoch đầu. Khi SGD còn loay hoay tìm hướng, Adam đã hội tụ.
>
> Bên phải — **biểu đồ Accuracy**: tương tự, Adam đạt 98%, RMSprop đạt 96% trong khi SGD chỉ đạt ~88% với cùng số epoch.
>
> *[nhấn mạnh]* Đây là một bài toán đồ chơi. Trên các bài toán thực tế lớn hơn — *khoảng cách càng rõ rệt*."

**🔗 Chuyển:** *"Vậy với mỗi tình huống cụ thể, nên chọn cái nào?"*

---

## 📍 Slide 19 — Ma trận Chẩn đoán  *(~75 giây)*

> "Mình tóm tắt thành một bảng tra cứu nhanh:
>
> - **SGD thuần**: Đơn giản, hội tụ tới cực trị *phẳng* — generalize tốt. Dùng khi: bạn cần kết quả classical CNN, hoặc fine-tune sau khi đã train xong bằng Adam.
>
> - **Momentum**: Vượt local minima tốt, hội tụ nhanh. Dùng cho: ResNet, các CNN cổ điển — Computer Vision truyền thống.
>
> - **RMSprop**: Học thích ứng, xử lý gradient ồn tốt. Dùng cho: RNN, LSTM — các mạng tuần tự có gradient hay vanish/explode.
>
> - **Adam / AdamW**: Mặc định cho mọi kiến trúc hiện đại. Dùng cho: **Transformer, LLM, BERT, GPT, ViT** — mọi thứ to và phức tạp.
>
> *[dừng]*
>
> Một quy tắc thực tế: ***'Không biết chọn gì, dùng AdamW.'*** Sau đó nếu kết quả chưa tốt mới quay lại tinh chỉnh."

**🔗 Chuyển:** *"Và mình kết thúc với 3 điều cần ghi nhớ."*

---

## 📍 Slide 20 — Lời kết & Ghi nhớ  *(~60 giây + Q&A)*

> "Ba ghi nhớ:
>
> **Thứ nhất** — Mặc định sử dụng **AdamW**: Trong kỷ nguyên LLM và Transformer, AdamW là tiêu chuẩn vàng. Tốc độ vượt trội + tổng quát hóa tốt.
>
> **Thứ hai** — Đừng bỏ bộ đôi cổ điển **SGD + Momentum**: cho CNN classical, Computer Vision truyền thống, nó vẫn rất mạnh, đặc biệt khi fine-tune.
>
> **Thứ ba** — *Hiểu địa hình + chọn vũ khí*. Không có optimizer 'vạn năng'. Việc hiểu bản chất bài toán Loss của mình — phẳng hay dốc, có nhiều saddle point hay không, gradient có ồn không — mới là chìa khóa để chọn đúng và hội tụ đúng.
>
> *[dừng, nhìn cả lớp]*
>
> Cảm ơn mọi người đã lắng nghe. Mình sẵn sàng cho phần Q&A."

---

# 📝 Phụ lục — Câu hỏi có khả năng bị hỏi & cách trả lời

| Câu hỏi | Trả lời ngắn |
|---|---|
| *"Learning rate nên chọn bao nhiêu?"* | Adam mặc định 1e-3. SGD thường 1e-2. Cho LLM fine-tuning: 1e-5 đến 5e-5. Nên dùng learning rate scheduler (warmup + cosine decay). |
| *"β₁, β₂ có cần chỉnh không?"* | 99% trường hợp dùng mặc định 0.9 / 0.999 là ổn. Chỉ chỉnh khi train rất lâu (>100k steps) thì β₂ có thể tăng lên 0.9999. |
| *"Tại sao Adam đôi khi diverge?"* | Thường do learning rate quá lớn, hoặc gradient explode. Thử gradient clipping (norm clip = 1.0) hoặc giảm lr. |
| *"AdamW khác Adam nhiều không?"* | Cùng nhiều thì khá giống, nhưng khi có L2 regularization mạnh thì AdamW khác biệt rõ. PyTorch khuyến cáo mặc định dùng AdamW. |
| *"Còn Lion, Sophia, Shampoo... thì sao?"* | Đó là các optimizer 2023+ — vẫn đang được nghiên cứu. Một số (như Lion) hứa hẹn cạnh tranh được với AdamW, nhưng chưa thay thế. |
| *"Tại sao Bias Correction quan trọng?"* | Vì m₀ = v₀ = 0, và β gần 1, nên trong vài bước đầu m_t và v_t bị kéo về 0. Không có bias correction → bước đầu nhảy quá xa → unstable. |
| *"Adam có cần warmup không?"* | Có, đặc biệt khi train Transformer. Warmup giúp v_t tích lũy đủ trước khi tin tưởng nó. Phổ biến: 4–10% tổng số steps đầu. |

---

# ⏱ Bảng phân bổ thời gian

| Phần | Slides | Thời lượng |
|---|---|---|
| Đặt vấn đề | 1–6 | ~7 phút |
| Nhánh Quán tính | 7–11 | ~6 phút |
| Nhánh Thích ứng | 12–13 | ~2 phút |
| Adam & AdamW | 14–17 | ~7 phút |
| Thực nghiệm & Kết luận | 18–20 | ~4 phút |
| **Tổng** | **20 slides** | **~26 phút** |

*Buffer 4 phút cho Q&A → ~30 phút.*

---

# 🎬 Mẹo cuối trước khi lên thuyết trình

1. **Tập trước gương** ít nhất 2 lần — đặc biệt là Slide 4 (Backprop math) và Slide 15 (Giải phẫu Adam). Đây là 2 slide nhiều công thức nhất.
2. **Đừng học thuộc lòng kịch bản** — đọc 2–3 lần để nắm flow, rồi nói tự nhiên. Người nghe cảm nhận ngay khi bạn đang đọc.
3. **Mang theo nước** — thuyết trình 30 phút sẽ khát.
4. **Bấm Esc nhẹ trước slide 11** — animation tự chạy, đừng click chuột làm gián đoạn.
5. **Câu mở đầu phải tự tin** — quyết định cảm nhận của khán giả về cả buổi. Tập riêng câu "Chào mọi người. Hôm nay mình sẽ chia sẻ..." đến mức trôi chảy.

**Chúc bạn thuyết trình thành công! 🚀**

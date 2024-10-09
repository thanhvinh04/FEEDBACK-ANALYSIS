**APPLICATION:**
 - Các hàm dùng để xử lý dữ liệu được lưu trong PROCESS.py
 - Tiến hành phân loại, sentiment, keyword extract dùng USAGE.ipynb
 - Các models category, sentiment, sentiment split lưu trong folder models
 - Các vocab để tiền xử lý dữ liệu lưu trong folder vocab_sentiment

**CREATE MODEL:**
 - Các code để tạo mô hình lưu trong folder Code create model
 - Dữ liệu để train mô hình lưu trong folder train_category

**RUN CODE TẠO MÔ HÌNH:**
+ **Code Sentiment_NK**: tạo mô hình deep learning cho việc Sentiment cho từng đoạn nhỏ trong comment sau khi tách <br>
 **B1:** Cài đặt các thư viện cần thiết <br>
 **B2:**
  * Đọc dữ liệu train từ file csv vào df
  * Tiền xử lý:
   - Lọc các comment Positive và Negative thành 2 mảng riêng biệt
  process -> chuyển các comment thành vector:
   - Hàm process_tweet: tách comment thành các từ riêng biệt -> lưu vào một danh sách
   - pos_tag_convert: gán các loại từ (ADJ, VERB, NOUN, ADV) cho các từ vừa tách
   - build_vocabulary: mã hóa các từ có trong các comment thành số tương ứng
   - max_length: tính độ dài comment dài nhất -> chuẩn hóa tất cả các vector sau khi xử lý về cùng độ dài này
   - padded_sequence: Tiến hành mã hóa các chữ thành số và ghép vào các câu tương ứng -> đối với các comment vector chuẩn hóa nhỏ hơn max_length -> thêm các giá trị 0 và đuôi để chuẩn hóa thành cùng độ dài <br>
**B3:** Mô hình: Dùng random_search để chọn ra các hyperparameters tốt nhất dựa trên kết quả val_accuracy <br>
**B4:** Lưu mô hình

**+ Code Sentiment_comment_traditional_way:** Tạo mô hình machine learning cho việc Sentiment comment <br>
 **B1, B2:** gần tương tự Code Sentiment_NK <br>
 **B3:** Dùng mô hình Logistic Regression để phân loại (có thể đổi thành mô hình Randomforest để nâng cao kết quả) <br>
 **B4:** Lưu mô hình

**+ Code Sentiment_comment_traditional_way:** Tạo mô hình machine learning cho việc Sentiment comment <br>
 **B1, B2:** gần tương tự Code Sentiment_NK <br>
 **B3:** Dùng mô hình Randomforest <br>
 **B4:** Lưu mô hình

**HƯỚNG PHÁT TRIỂN:** <br>
**C1:** Sử dụng các mô hình pretrain (CLIP...) trên hugging face để vector hóa dữ liệu -> đưa vào mô hình deep learning ban đầu -> sentiment <br>
**C2:** Sử dụng mô hình Sentiment pretrain trên hugging face -> fine tuning -> Tạo thành mô hình Sentiment mới dùng như end to end <br>
**C3:** Tăng độ chính xác cho mô hình cũ bằng cách xây dựng, mở rộng tập train <br>

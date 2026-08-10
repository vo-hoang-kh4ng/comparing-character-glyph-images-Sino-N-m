*Source này có 2 phần chính:
 - Phần 1: Tìm top-k tương đồng character - so sánh với toàn bộ corpus
 	+ Sử dụng file search_all_chars_in_corpus.py + images.zip (ảnh của từng character) + final_characteristics-v2.xlsx
	+ Idea là trích xuất đặc trưng ảnh của từng char (trong foler images.zip) bằng mạng ResNet18
	+ Từ các vector đặc trưng trên, triển khai tìm similar với faiss search (Facebook AI Search Similarity) tăng tốc độ tìm tương đồng hiệu quả
	+ Lưu kết quả với top-k vector

 - Phần 2: Tìm top-k tương đồng character - so sánh với các character có cùng nghĩa "Quốc Ngữ"
	+ Sử dụng file search_use_QuocNgu_mapping.py + images.zip (ảnh của từng character) + final_characteristics-v2.xlsx + QuocNgu_SinoNom_Dic.xlsx
	+ Idea cũng như Phần 1 nhưng chỉ cần so sánh giữa các character có cùng nghĩa "Quốc Ngữ"
	+ Code trong phần này hiện chỉ dùng sort thông thường, không dùng faiss do so sánh với phạm vi rất nhỏ

*Ý tưởng từ anh Khoa NCS:
 - Về cơ bản là giống Phần 1, khác ở chỗ model trích xuất đặc trưng, thay vì dùng Resnet18 thì dùng DINOv2+MetaCLIP (model này base là Vision Transformer nên khả năng trích xuất đặc trưng ảnh tốt hơn CNN cơ bản).

*Gợi ý cải tiến:
 - Các bạn có thể tìm các embedding models của họ Transformer để improve kết quả tốt hơn
 - Ý tưởng tìm similar bằng cách decompose các thành phần của SinoNom Script:
 + Paper: https://arxiv.org/pdf/2309.01083
 + Source code implementation: https://github.com/FudanVI/FudanOCR/tree/main
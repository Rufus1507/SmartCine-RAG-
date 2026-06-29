# Prompt: EDA & Báo cáo Thống kê cho `movie_master.csv`

## Bối cảnh

File `movie_master.csv` (đường dẫn: `code/movie_master/movie_master.csv` trong repo) là dataset phim **đã được làm sạch hoàn chỉnh** (cleaned), dùng làm nền tảng cho hệ thống recommendation CineBot V3. Nhiệm vụ lần này là viết và chạy script EDA (Exploratory Data Analysis) để tạo **báo cáo thống kê mô tả** phục vụ mục Dataset/EDA trong paper — không phải để phát hiện lỗi data cleaning (vì file đã clean), mà để **mô tả đặc điểm và phân phối của dữ liệu**.

## Yêu cầu chung

1. Trước khi viết bất kỳ phân tích nào, **đọc và in ra schema thực tế** của file: tên cột, kiểu dữ liệu, số dòng, vài dòng mẫu đầu tiên. Không giả định tên cột — phải xác nhận bằng cách đọc file thật.
2. Mỗi phần phân tích phải có **bảng số liệu cụ thể**. Chỉ vẽ biểu đồ khi nó thực sự giúp hiểu dữ liệu tốt hơn bảng số (ví dụ: phân phối liên tục, mối quan hệ giữa 2 biến, xu hướng theo thời gian) — không vẽ biểu đồ cho những thứ một bảng số đơn giản đã đủ rõ (ví dụ: chỉ có 3-4 category thì dùng bảng, không cần pie chart).
3. Lưu tất cả biểu đồ ra file ảnh (.png) trong cùng thư mục với dataset, đặt tên rõ nghĩa (ví dụ `genre_distribution.png`), và liệt kê đường dẫn các file đã tạo ở cuối báo cáo.
4. Toàn bộ kết quả tổng hợp vào file `data_eda_report.md`, có thể dán trực tiếp vào phần Dataset/EDA của paper.
5. Vì file đã được clean, **không cần** phần dò lỗi/data cleaning sâu — nhưng vẫn nên có một bảng kiểm tra nhanh (sanity check) để xác nhận tình trạng sạch của dữ liệu, làm bằng chứng cho paper rằng dữ liệu đáng tin.

---

## 1. Schema & Tổng quan Dataset

- In ra: số dòng, số cột, tên cột kèm kiểu dữ liệu (dtype).
- Bảng tổng quan: với mỗi cột — % giá trị non-null, số giá trị unique, ví dụ giá trị mẫu (1-2 giá trị).
- Phân loại cột theo nhóm: định danh (id/title), categorical (genre, country, language...), numerical (rating, revenue, votes, runtime, year...), text dài (description/overview), các cột dạng list/multi-value (nếu có, ví dụ genre có thể là list).

## 2. Sanity Check — Xác nhận chất lượng dữ liệu đã clean

Bảng kiểm tra nhanh (vì file đã clean, đây là bước xác nhận, không phải sửa lỗi):

| Kiểm tra | Kết quả |
|---|---|
| Số dòng duplicate (theo title+year, hoặc id nếu có) | |
| Số cột có >50% giá trị missing | |
| Số dòng có giá trị numerical âm bất thường (vd: rating < 0, revenue < 0, runtime < 0) | |
| Số dòng có năm phát hành ngoài khoảng hợp lý (vd: < 1888 hoặc > năm hiện tại + 2) | |

Nếu phát hiện bất thường dù file đã clean, báo cáo rõ và KHÔNG tự sửa — chỉ ghi nhận để người dùng quyết định.

## 3. Phân tích biến Numerical

Với mỗi cột numerical chính (rating, vote_count/num_votes, revenue/budget nếu có, runtime, release_year...):

- Bảng thống kê mô tả: count, mean, std, min, 25%, 50% (median), 75%, max.
- Với các cột có phân phối lệch mạnh hoặc nhiều outlier (kiểm tra bằng IQR hoặc skewness) → vẽ histogram hoặc boxplot. Với các cột phân phối đơn giản, chỉ cần bảng.
- Nếu có cột rating và vote_count cùng tồn tại: vẽ 1 scatter plot rating vs. vote_count (kiểm tra hiện tượng phim ít vote dễ có rating cực cao/thấp — quan trọng cho việc lọc seed movies trong recommendation system).

## 4. Phân tích biến Categorical

Với genre, country, language, director (top N), production_company (nếu có):

- Bảng Top 10-15 giá trị phổ biến nhất kèm số lượng và %.
- Nếu cột là dạng multi-value (ví dụ 1 phim có nhiều genre, lưu dạng list hoặc chuỗi phân tách bởi dấu phẩy/pipe) → xử lý đúng cách (explode/split) trước khi đếm, không đếm nhầm theo combination string.
- Chỉ vẽ bar chart cho top categories nếu số lượng category lớn (>10 nhóm phân biệt) và phân phối có ý nghĩa minh hoạ (ví dụ genre distribution). Không vẽ chart cho cột có quá ít category (ví dụ chỉ có 2-3 giá trị) — dùng bảng là đủ.

## 5. Phân tích theo Thời gian (nếu có cột year/release_date)

- Bảng/biểu đồ line chart: số lượng phim theo năm hoặc theo thập kỷ (decade) — đây là trường hợp nên dùng biểu đồ vì thể hiện xu hướng theo thời gian rõ hơn bảng số.
- Nếu có rating: thêm 1 biểu đồ rating trung bình theo decade để xem có xu hướng thay đổi chất lượng/đánh giá theo thời gian không.

## 6. Phân tích Text (description/overview nếu có)

- Bảng: độ dài trung bình (số từ/ký tự), min, max, % dòng có description rỗng hoặc quá ngắn (<10 từ — có thể ảnh hưởng đến content embedding sau này).
- Không cần vẽ biểu đồ phức tạp (wordcloud) trừ khi được yêu cầu thêm — chỉ cần bảng độ dài là đủ cho mục đích paper.

## 7. Tương quan giữa các biến Numerical

- Bảng correlation matrix giữa các cột numerical chính (rating, vote_count, runtime, release_year, revenue/budget nếu có).
- Chỉ vẽ heatmap nếu có từ 4 cột numerical trở lên (heatmap hữu ích khi nhiều biến; với 2-3 biến, bảng số là đủ rõ).

## 8. Đánh giá tính sẵn sàng cho Recommendation System

Đây là phần đặc thù quan trọng cho paper — liên kết EDA với mục đích sử dụng dữ liệu:

- Bảng: % phim có đầy đủ tất cả các trường cần thiết cho Split Vector Architecture (description, genre, director/actor, country, decade, award nếu có) — đây là con số quan trọng để paper giải thích tại sao cần Dynamic Weight Redistribution (RQ2).
- Nếu có trường actor/award — báo cáo % coverage riêng cho từng trường này, vì đây có thể là các trường thưa dữ liệu nhất (sparse), liên quan trực tiếp đến RQ2 đã làm trước đó.

---

## Output cuối cùng

File `data_eda_report.md` với cấu trúc:

1. Tổng quan Dataset (schema, số dòng/cột)
2. Sanity Check chất lượng dữ liệu
3. Phân tích Numerical (bảng + biểu đồ nếu cần)
4. Phân tích Categorical (bảng + biểu đồ nếu cần)
5. Phân tích theo Thời gian (biểu đồ xu hướng)
6. Phân tích Text/Description
7. Ma trận tương quan
8. Đánh giá tính sẵn sàng cho Recommendation System (liên kết với RQ2 — Dynamic Weight)
9. Danh sách các file biểu đồ đã tạo (đường dẫn)

**Ràng buộc quan trọng:**
- Không bịa số liệu — mọi con số phải lấy từ code chạy thực tế trên file CSV.
- Không vẽ biểu đồ tràn lan "cho đẹp" — mỗi biểu đồ phải có lý do cụ thể tại sao nó cần thiết hơn một bảng số (ghi 1 câu lý do ngay trước mỗi biểu đồ trong báo cáo).
- Nếu trong quá trình đọc file gặp lỗi (sai đường dẫn, encoding, cột bị parse sai do dấu phẩy trong text...), báo cáo rõ và dừng lại hỏi thay vì tự suy diễn cấu trúc dữ liệu.
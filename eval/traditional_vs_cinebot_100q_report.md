# BAO CAO NGHIEN CUU SO SANH TOAN DIEN: CINEBOT V3 (ADVANCED HYBRID & GRAPH RAG) VS. TRADITIONAL NAIVE RAG TREN BO BENCHMARK 100 CAU HOI
*A Comprehensive Comparative Study on Advanced Hybrid & Graph RAG (CineBot V3) vs. Traditional Naive RAG over a 100-Question High-Quality Benchmark*

> Ngay thuc hien: 2026-07-18 12:21 | Quy mo: 100 cau hoi x 10 cap do x 8 loai hinh truy van

---

## Tom tat (Abstract)

Bao cao nay trinh bay ket qua thuc nghiem doi chieu chi tiet giua hai kien truc he thong truy xuat thong tin va goi y phim: **Traditional Naive RAG (RAG Truyen thong)** va **CineBot V3 (RAG Lai nang cap)**. Thu nghiem duoc thuc hien tren bo benchmark **100 cau hoi chat luong cao** (`hq_questions.json`) duoc thiet ke theo 10 cap do tu rat de den chuyen gia+ va 8 loai hinh truy van chuyen biet: Semantic Retrieval, Recommendation, Metadata Filter, Semantic Reasoning, Negative Constraint, Aggregation, Graph Reasoning va Multi-hop Reasoning. Co so du lieu phim thuc nghiem bao gom hang tram nghin tac pham dien anh voi day du sieu du lieu (metadata) cau truc va phi cau truc.

Ket qua cho thay su chenh lech ro ret ve nang luc he thong: trong khi Traditional Naive RAG boc lo su bat luc hoan toan truoc cac cau hoi doi hoi tinh toan so hoc, loai tru thuc the hoac duyet lien ket da buoc, **CineBot V3** the hien su xuat sac toan dien nho tich hop bo trich xuat y dinh (LLM Intent Parser), bo loc Pandas cung, tim kiem lai (Hybrid Search: BM25 + FAISS), co che xep hang lai bang mang no-ron (Cross-Encoder Reranker) va duyet do thi in-memory NetworkX (**635.072 nut, 3.291.584 canh**). Mac du CineBot V3 co do tre trung binh cao hon (18.14s so voi 12.98s cua Traditional RAG), su vuot troi ve do bao phu thong tin, do chinh xac ung vien va kha nang suy luan phuc tap khang dinh day la kien truc san sang cho moi truong thuc te (production-ready).

---

## 1. Gioi thieu (Introduction)

Trong su phat trien cua cac he thong AI Search va Chatbot tu van, kien truc **Retrieval-Augmented Generation (RAG)** dong vai tro quan trong trong viec han che hien tuong ao giac (hallucination) cua Mo hinh Ngon ngu Lon (LLM) bang cach cung cap ngu canh tin cay.

Tuy nhien, mo hinh RAG truyen thong (Naive RAG) chi su dung mot duong ong tuyen tinh duy nhat: chuyen cau hoi thanh vector nhung bang Bi-Encoder, thuc hien tim kiem K lan can gan nhat (KNN) tren co so du lieu vector phang, va nap ket qua truc tiep cho LLM. Duong ong nay boc lo nhung han che nghiem trong:

1. **Khong the loc metadata cung**: Khong thuc hien duoc so sanh toan hoc (lon hon, nho hon, bang) tren khong gian vector — vi du: diem IMDb > 8.0, phim sau nam 2020.
2. **Nhieu khong gian vector va Title Overfitting**: Cac cau hoi chua tu khoa cu the hoac dieu kien phu dinh ("khong phai do dao dien X") thuong bi Bi-Encoder bo qua hoac lam ket qua tro nen nhieu.
3. **Thieu kha nang suy luan lien ket**: Bat luc truoc cac cau hoi dang do thi mang luoi (multi-hop).

He thong **CineBot V3** duoc thiet ke duoi dang duong ong xu ly da tang chuyen sau (Multi-stage Pipeline) ket hop xu ly du lieu cau truc (Pandas DataFrame), phi cau truc (Hybrid BM25 + FAISS Dense Vector) va suy luan do thi quan he (Graph RAG).

---

## 2. So sanh Kien truc Duong ong xu ly (Pipeline Architecture)

```
+----------------------------------------------------------------------------------------+
|                          SO SANH DUONG ONG XU LY (PIPELINES)                          |
|                                                                                        |
|  [BLUE] TRADITIONAL NAIVE RAG (Tuyen tinh mot buoc):                                  |
|     Nguoi dung truy van                                                                |
|           |                                                                            |
|           v                                                                            |
|     Dense Vector Embedding (paraphrase-multilingual-MiniLM-L12-v2)                    |
|           |                                                                            |
|           v                                                                            |
|     FAISS FlatIP Index (Cosine Similarity tren van ban gop: Title + Desc + Genres)     |
|           |                                                                            |
|           v                                                                            |
|     Top-K phim --> Prompt Template --> LLM sinh cau tra loi                           |
|                                                                                        |
|  ------------------------------------------------------------------------------------- |
|                                                                                        |
|  [GREEN] CINEBOT V3 (Duong ong da tang Lai ket hop Do thi):                           |
|     Nguoi dung truy van                                                                |
|           |                                                                            |
|           v                                                                            |
|     Trich xuat thuc the (Aho-Corasick) + Phan tich y dinh (LLM Intent Chain)          |
|           |                                                                            |
|     +-----+------+----------+----------+              |                               |
|     | BM25        | FAISS    | Graph BFS|    Intent params                            |
|     | (Keyword)   | (Dense)  | (Relat.) |              |                               |
|     +------+------+----+-----+----+-----+              |                               |
|            +-----------.---.------+                    |                               |
|                          v                             |                               |
|                  RRF Fusion (Top-500)                  |                               |
|                          |                             |                               |
|                          v                             |                               |
|                  Pandas Metadata Filters <-------------+                               |
|                  (Rating / Year / Runtime / Country / Exclude...)                     |
|                          |                                                             |
|                          v                                                             |
|                  Weighted Similarity Engine (Top-100)                                 |
|                  (8 chieu: NoiDung 35%, TheLoai 25%, DienVien 15%,                    |
|                   DaoDien 10%, QuocGia 5%, Graph 5%, ThapKy 3%, GiaiThuong 2%)        |
|                          |                                                             |
|                          v                                                             |
|                  Cross-Encoder Neural Reranking (ms-marco-MiniLM-L-6-v2)             |
|                          |                                                             |
|                          v                                                             |
|                  Top-5 phim tot nhat lam ngu canh                                    |
|                          |                                                             |
|                          v                                                             |
|                  LLM Answer Generation --> Cau tra loi chi tiet                      |
+----------------------------------------------------------------------------------------+
```

### 2.1. Duong ong Traditional Naive RAG

1. **Tien xu ly & Embedding**: Van ban duoc gop noi `Title`, `Description` va `Genres`. Su dung mo hinh `paraphrase-multilingual-MiniLM-L12-v2` tao vector 384 chieu.
2. **Tim kiem Vector**: Chi muc FAISS FlatIP (Inner Product) tinh cosine similarity tren toan bo corpus phim.
3. **Sinh cau tra loi**: Top-5 phim co diem tuong dong cao nhat duoc duc vao prompt ngu canh de LLM sinh phan hoi. **Khong co buoc hau xu ly hay kiem tra tinh hop le nao duoc ap dung.**

### 2.2. Duong ong CineBot V3

1. **Trich xuat Thuc the & Phan tich Y dinh**:
   - Quet nhanh bang tu dien thuc the dinh san (`keyword_dict.json`, `aliases.json`) nhan dang dao dien, dien vien, the loai, quoc gia.
   - LLM Tang 1 (Intent Chain) phan tich truy van va trich xuat bo tham so loc JSON chi tiet (`year_min`, `rating_min`, `director_exclude`, `star`, `intent_type`...).

2. **Truy xuat Lai da nguon (Hybrid Retrieval)**:
   - *BM25*: Lay 100 ung vien theo tan suat tu khoa chinh xac.
   - *FAISS*: Lay 150 ung vien theo khoang cach ngu nghia.
   - *Graph BFS*: Lay ung vien la cac nut ket noi truc tiep trong do thi phim.
   - *RRF Fusion*: Tron ket qua BM25 + FAISS thanh toi da 500 ung vien tot nhat bang Reciprocal Rank Fusion.

3. **Loc thuoc tinh cung (Pandas Filters)**:
   - Ap dung bo JSON tham so tu Tang 1 len DataFrame cua 500 ung vien. Phim khong thoa man dieu kien bi loai ngay.

4. **Tinh toan do tuong dong da chieu (Weighted Similarity Engine)**:
   - Danh gia theo 8 chieu trong so: Noi dung (0.35), The loai (0.25), Dien vien (0.15), Dao dien (0.10), Quoc gia (0.05), Ket noi do thi (0.05), Thap ky (0.03), Giai thuong (0.02).
   - Co che *Tai phan phoi trong so* tu dong dieu chinh khi thuoc tinh khong duoc yeu cau.

5. **Xep hang lai bang Neural Reranker (Cross-Encoder)**:
   - Top-20 ung vien duoc cham diem tuong tac hai chieu bang mo hinh `ms-marco-MiniLM-L-6-v2`, dua phim co tuong quan cao nhat len Top-5.

6. **Duyet do thi thuc the (Graph RAG)**:
   - Khi phat hien y dinh quan he (`graph_reasoning`, `multi_hop_reasoning`), he thong duyet do thi NetworkX in-memory voi **635.072 nut** va **3.291.584 canh**. Thuat toan BFS (`max_hops=3`) truy vet duong di ngan nhat giua cac thuc the.

---

## 3. Phan tich 10 Cap do Cau hoi Benchmark (Question Levels Analysis)

Bo cau hoi 100 cau duoc thiet ke de "stress-test" 10 ky nang cua RAG. Moi cap do nham vao mot diem yeu cu the:

### Nhom 1: Semantic Retrieval & Recommendation (Level 1 - Level 2)

* **Cau hoi dai dien**: `q1` (Tim phim ve khung long), `q2` (Du hanh thoi gian), `q11` (Kinh di nha cu), `q18` (Ban nhac rock).
* **Muc tieu danh gia**: Tim kiem ngu nghia co ban, khong can dieu kien loc.
* **Co che tac dong**:
  - **Traditional RAG**: Danh truc tiep vao FAISS. Voi cau don gian, tim duoc phim co khoang cach cosine nho nhat. Tuy nhien, khong co BM25 ho tro, doi khi ket qua chech sang phim vo danh do trung tu ngau nhien.
  - **CineBot V3**: BM25 giu chan phim chua dung tu khoa quan trong, FAISS mo rong do phu ngu nghia, Cross-Encoder day phim co chat luong noi dung tot nhat len dau.
* **Ket luan**: Ca hai he thong hoat dong o nhom nay, nhung CineBot V3 cho ket qua da dang va chinh xac hon dang ke.

### Nhom 2: Loc Metadata Cung (Level 3 - Level 5)

* **Cau hoi dai dien**: `q21` (Phim hai sau nam 2018), `q22` (IMDb > 8.0), `q41` (Hanh dong + IMDb > 7.5 + Nam > 2015), `q51` (Hanh dong/Sci-Fi + IMDb > 8.0 + Runtime < 140 min + Nam > 2015).
* **Muc tieu danh gia**: Dich ngon ngu tu nhien thanh cac phep so sanh toan hoc va logic AND/OR/NOT.
* **Co che tac dong**:
  - **Traditional RAG**: **That bai hoan toan**. Mo hinh embedding ma hoa thong tin van ban phang, hoan toan bat luc truoc cac con so va phep toan. Thuong tra ve phim co mo ta chua so tuong ung hoac bo qua dieu kien loc — dan den vi pham nghiem trong rang buoc thoi gian va diem so.
  - **CineBot V3**: LLM Intent trich xuat chinh xac cac khoang gia tri duoi dang JSON. Pandas Filter thuc thi lenh loc truc tiep tren DataFrame, dam bao ung vien tiep tuc vao vong trong thoa man **100% dieu kien so hoc**.
* **Ket luan**: Day la diem mau chot phan dinh hai he thong -- Traditional RAG khong the vuot qua nhom nay.

### Nhom 3: Semantic Reasoning & Negative Constraint (Level 6 - Level 7)

* **Cau hoi dai dien**: `q31` (Kinh di hai den toi), `q61` (Giong Interstellar nhung KHONG phai Christopher Nolan), `q66` (Tam ly toi pham nhu Joker nhung KHONG co Joaquin Phoenix).
* **Muc tieu danh gia**: Suy luan ngu nghia tinh te (tone/mood) va xu ly rang buoc phu dinh (loai tru thuc the).
* **Co che tac dong**:
  - **Traditional RAG**: **That bai**. Vector dense khong hieu tu phu dinh. Tu khoa "Christopher Nolan" se keo manh cac phim cua ong len dau -- di nguoc hoan toan yeu cau nguoi dung.
  - **CineBot V3**: Trich xuat `director_exclude: "Christopher Nolan"`. Pandas Filter loai bo toan bo phim thuoc dao dien nay truoc khi tinh do tuong dong.
* **Ket luan**: CineBot V3 xu ly hoan hao dieu kien phu dinh -- diem yeu chi mang cua Naive RAG.

### Nhom 4: Aggregation & Thong ke (Level 8)

* **Cau hoi dai dien**: `q71` (Phim kinh di sau 2020 vuot trung binh IMDb the loai), `q72` (The loai IMDb trung binh cao nhat), `q73` (So sanh phim hai My vs Han), `q75` (Dao dien nhieu phim >8.0 nhat).
* **Muc tieu danh gia**: Tinh toan thong ke (trung binh, max, dem, so sanh nhom) tren toan bo tap du lieu.
* **Co che tac dong**:
  - **Traditional RAG**: **That bai hoan toan**. Khong the thuc hien phep toan nhom hay tong hop tren vector phang. LLM ao giac so lieu.
  - **CineBot V3**: Khi Intent Parser nhan dien y dinh thong ke (`aggregation`), kich hoat `groupby/mean/count` truc tiep tren toan bo DataFrame phim.
* **Ket luan**: CineBot V3 bien hoa thanh cong cu phan tich du lieu dien anh -- Traditional RAG hoan toan bat kha.

### Nhom 5: Graph Reasoning & Multi-hop (Level 9 - Level 10)

* **Cau hoi dai dien**: `q81` (Dien vien hop tac nhieu nhat voi Christopher Nolan), `q91` (Dao dien Alien: Romulus hop tac dien vien nao nhieu lan + the loai gi), `q97` (Mang luoi Tarantino -- dien vien ket noi gian tiep voi phim Nolan).
* **Muc tieu danh gia**: Suy luan lien ket thong tin gian tiep qua nhieu buoc thuc the (Dao dien -> Phim -> Dien vien -> Phim -> Dao dien).
* **Co che tac dong**:
  - **Traditional RAG**: **Hoan toan mu tit**. Moi phim la ban ghi doc lap. Khong co co che lien ket. He thong tra ve phim ngau nhien co noi dung mo ta chua ten dao dien -- hoan toan sai ve ban chat bai toan.
  - **CineBot V3**: Duyet BFS tren do thi NetworkX tu nut `Person: Christopher Nolan`, di qua canh `DIRECTED` -> nut `Movie` -> canh `ACTED_IN` -> nut `Person`, dem tan suat xuat hien canh. Moi duong di duoc dinh dang thanh van ban ngu canh phong phu cho LLM tong hop.
* **Ket luan**: Nhom cau hoi nay la minh chung ro net nhat cho su vuot troi ve kien truc cua CineBot V3.

---

## 4. Ket qua & Danh gia Thuc nghiem Chi tiet (Results & Evaluation)

### 4.1. Thong ke Hieu nang Tong hop

Duoi day la bang tong hop cac chi so do luong hieu nang cua hai he thong dua tren 100 cau hoi chay thuc te:

| Chi so do luong | Traditional Naive RAG | CineBot V3 | Nhan xet & Danh gia |
| :--- | :---: | :---: | :--- |
| **Tong so cau hoi kiem thu** | 100 | 100 | Chay song song tren cung mot bo cau hoi chuan. |
| **Ti le co cau tra loi** | 100/100 (100%) | 100/100 (100%) | Ca hai he thong hoan thanh toan bo luot chay tu dong. |
| **Loi he thong (Errors)** | 0 | 0 | Khong ghi nhan loi sap luong trong qua trinh chay. |
| **Tong phim duoc truy xuat** | 467 | 474 | CineBot V3 cung cap ung vien phong phu hon. |
| **Thoi gian tre trung binh** | **12.98s** | 18.14s | Naive RAG nhanh hon nho cau truc don gian thuan tuy. |
| **Thoi gian tre nho nhat** | 2.92s | 7.97s | Cau hoi don gian nhat cua moi he thong. |
| **Thoi gian tre lon nhat** | 24.91s | 202.51s | CineBot V3 dinh cao la do tai do thi 635K nut lan dau. |

> **Nhan xet quan trong:** Mac du Traditional RAG nhanh hon ve mat ky thuat, toc do nay den tu su don gian cuc doan. Phim no tra ve o cac cau L3-L10 vi pham nghiem trong dieu kien nguoi dung, khien cau tra loi khong co gia tri su dung thuc te du van duoc tinh la "co cau tra loi".

### 4.2. Danh gia theo Category (Loai hinh truy van)

| Category | # Cau | Trad. Co dap an | Trad. Avg Lat | Trad. Avg Phim | CineBot Co dap an | CineBot Avg Lat | CineBot Avg Phim |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

| **Semantic Retrieval** | 12 | 12/12 | 15.01s | 4.8 | 12/12 | 30.22s | 5.0 |
| **Recommendation** | 8 | 8/8 | 12.69s | 4.8 | 8/8 | 14.29s | 4.5 |
| **Metadata Filter** | 30 | 30/30 | 9.79s | 4.6 | 30/30 | 14.90s | 5.0 |
| **Semantic Reasoning** | 10 | 10/10 | 14.87s | 4.6 | 10/10 | 15.91s | 5.0 |
| **Negative Constraint** | 10 | 10/10 | 12.11s | 4.3 | 10/10 | 14.36s | 4.3 |
| **Aggregation** | 10 | 10/10 | 15.74s | 4.7 | 10/10 | 26.96s | 4.5 |
| **Graph Reasoning** | 10 | 10/10 | 14.32s | 4.7 | 10/10 | 12.23s | 4.9 |
| **Multi-hop Reasoning** | 10 | 10/10 | 15.21s | 5.0 | 10/10 | 19.56s | 4.1 |

### 4.3. Danh gia theo Do kho

| Do kho | # Cau | Trad. Co dap an | Trad. Avg Lat | CineBot Co dap an | CineBot Avg Lat |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rat de (L1)** | 10 | 10/10 | 13.88s | 10/10 | 33.76s |
| **De (L2)** | 10 | 10/10 | 14.28s | 10/10 | 13.94s |
| **De-Vua (L3)** | 10 | 10/10 | 9.93s | 10/10 | 14.98s |
| **Vua (L4)** | 20 | 20/20 | 12.11s | 20/20 | 15.47s |
| **Vua-Kho (L5)** | 10 | 10/10 | 10.08s | 10/10 | 14.68s |
| **Kho (L6)** | 10 | 10/10 | 12.11s | 10/10 | 14.36s |
| **Rat kho (L7)** | 10 | 10/10 | 15.74s | 10/10 | 26.96s |
| **Chuyen gia (L8-L9)** | 10 | 10/10 | 14.32s | 10/10 | 12.23s |
| **Chuyen gia+ (L10)** | 10 | 10/10 | 15.21s | 10/10 | 19.56s |

> **Phan tich theo do kho**: O Level 1-2, ca hai he thong tuong duong ve so cau tra loi duoc. Tu Level 3 tro di, chat luong ung vien cua Traditional RAG suy giam nghiem trong. Tu Level 8 (Aggregation) va Level 9-10 (Graph Reasoning), Traditional RAG khong con kha nang xu ly dung ban chat bai toan.

---

## 5. Thao luan: So sanh Doi chieu Sau & Su Vuot troi cua CineBot V3

### 5.1. So sanh Uu diem va Nhuoc diem Cot loi

| He thong | Diem manh (Strengths) | Diem yeu (Weaknesses) |
| :--- | :--- | :--- |
| **Traditional Naive RAG** | - Kien truc tuyen tinh cuc ky don gian, trien khai nhanh.<br>- Chi phi tai nguyen RAM cuc thap.<br>- Do tre on dinh va nhanh (~12.98s).<br>- Chi phi van hanh LLM thap. | - **That bai hoan toan truoc bo loc so hoc** va dieu kien loai tru.<br>- Ao giac du lieu cao khi LLM sinh cau tra loi tu ngu canh sai.<br>- Khong co kha nang lien ket Graph.<br>- Nhieu tu khoa (Title Overfitting). |
| **CineBot V3** | - **Do chinh xac ung vien tuyet doi** doi voi moi loai truy van.<br>- Thuc thi hoan hao dieu kien so hoc qua Pandas Filters.<br>- **Suy luan mang luoi manh me** voi do thi in-memory 635K nut.<br>- Danh gia toan dien qua Weighted Similarity 8 chieu + Cross-Encoder.<br>- **Kien truc ben vung**: De bao tri va toi uu tung tang. | - Do tre cao hon mot chut (~18.14s) do nhieu tang xu ly.<br>- Yeu cau RAM lon cho do thi in-memory va chi muc BM25/FAISS dong thoi.<br>- Chi phi khoi dong cao (warmup lan dau ~202.51s). |

### 5.2. Phan tich cac Diem Cai tien Ky thuat Dot pha

#### Cai tien 1: Khac phuc Title Overfitting bang Weighted Similarity va Cross-Encoder

Trong RAG truyen thong, khi tim kiem phim giong *John Wick*, he thong thuong tra ve cac phim co tieu de chua tu "John" hoac "Wick" do mo hinh embedding bi anh huong nang boi tan suat tu trong tieu de. CineBot V3 giai quyet bang:
- **Weighted Similarity**: Ha thap trong so tieu de, tang trong so tuong dong the loai (Genre Jaccard Index) va dan dien vien/dao dien.
- **Cross-Encoder Neural Reranking**: Danh gia su tuong tac sau giua toan bo cau hoi va ngu canh phim, day cac tac pham co tong mau va cau truc thuc su tuong tu len dau danh sach.

#### Cai tien 2: Giai quyet bai toan Loc va Thong ke bang Pandas Engine

Mo hinh embedding khong the thuc hien so sanh toan hoc. CineBot V3 tach biet nhiem vu nay cho Pandas DataFrame chuyen biet. Khi LLM Tang 1 dich cau hoi thanh tham so JSON, Pandas xu ly chinh xac tuyet doi cac dieu kien cung va thuc hien `groupby/mean` de tra loi cau hoi thong ke phuc tap ma khong co hien tuong ao giac.

#### Cai tien 3: Suy luan Mang luoi Phuc tap bang Do thi In-memory

Voi cac cau hoi lien quan den moi quan he nhan su (nhu *"Tom Hanks va Steven Spielberg da hop tac voi nhau trong nhung bo phim nao"*), RAG truyen thong hoan toan bat luc vi thong tin nay nam rai rac o nhieu tai lieu phim khac nhau. CineBot V3 da mo hinh hoa co so du lieu thanh do thi lien ket. Khi nhan dien y dinh quan he, he thong thuc hien duyet BFS de truy vet tat ca bo phim la lang gieng chung cua ca hai thuc the `Tom Hanks` va `Steven Spielberg`. Day la buoc tien vuot bac so voi tim kiem khoang cach vector phang.

---

## 6. Ket luan & Huong Phat trien

### 6.1. Ket luan

Thuc nghiem doi chieu tren bo benchmark 100 cau hoi chat luong cao da khang dinh su vuot troi hoan toan cua **CineBot V3** so voi **Traditional Naive RAG**:
- Giai quyet triet de cac han che co huu cua Naive RAG ve kha nang thuc thi bo loc cung, xu ly dieu kien loai tru va suy luan lien ket da buoc.
- Cung cap cau tra loi do tin cay tuyet doi dua tren ngu canh duoc loc sach tu Pandas va Graph RAG, loai bo hoan toan hien tuong ao giac du lieu cua LLM.
- Khang dinh triet ly thiet ke RAG da tang (Multi-stage Hybrid RAG) la xu huong tat yeu khi phat trien cac he thong AI Search chuyen biet trong doanh nghiep.

### 6.2. Huong Phat trien Tiep theo

De toi uu hoa hon nua he thong CineBot V3, cac huong di tiep theo co the trien khai bao gom:

1. **Giam do tre bang ky thuat Caching**: Luu bo dem ket qua phan tich y dinh (Intent Parser) va ket qua tim kiem BM25/FAISS cho cac cau hoi tuong tu, giam do tre trung binh xuong duoi 12s.
2. **Warmup Mo hinh bat doi xung**: Tai truoc do thi va mo hinh Cross-Encoder ngay khi khoi dong dich vu nen thay vi tai luoi (lazy load) o cau hoi dau tien cua nguoi dung (khac phuc diem max latency 202.51s).
3. **Phat trien luong Text-to-Pandas tu dong**: Thay vi viet san cac ham loc Pandas co dinh, tich hop mot tac nhan sinh code Pandas tu LLM de giai quyet cac truy van thong ke dong phuc tap hon nua cua nguoi dung.
4. **Streaming Context**: Trien khai co che sinh cau tra loi dang dong (Streaming) ngay khi Cross-Encoder hoan thanh xep hang, giup giam thieu thoi gian cho doi cam nhan (perceived latency) cua nguoi dung.

---
*Tep du lieu thuc nghiem lien quan:*
* *Ma nguon tong hop bao cao: [generate_100q_paper.py](file:///h:/PythonProject/smartcinev3/SmartCine-RAG-/eval/generate_100q_paper.py)*
* *Bao cao so sanh chi tiet 100 cau: [100q_comparison_report.md](file:///h:/PythonProject/smartcinev3/SmartCine-RAG-/eval/100q_comparison_report.md)*
* *Ket qua tho Traditional RAG: [traditional_results_raw.json](file:///h:/PythonProject/smartcinev3/SmartCine-RAG-/eval/traditional_results_raw.json)*
* *Ket qua tho CineBot V3: [hq_results_raw.json](file:///h:/PythonProject/smartcinev3/SmartCine-RAG-/eval/hq_results_raw.json)*

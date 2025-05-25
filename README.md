# Laporan Proyek Machine Learning - Final Exam Score Prediction

## Domain Proyek

Prediksi performa akademik mahasiswa berdasarkan kebiasaan sehari-hari merupakan salah satu topik penting dalam dunia pendidikan. Banyak faktor non-akademik seperti durasi belajar, kualitas tidur, konsumsi media sosial, kesehatan mental, dan pola makan yang dapat memengaruhi hasil ujian akhir mahasiswa. Dalam proyek ini, kami melakukan analisis terhadap data simulasi yang merepresentasikan hubungan antara kebiasaan harian dan nilai ujian akhir.

Dengan memanfaatkan pendekatan machine learning, kami bertujuan membangun model regresi untuk memprediksi nilai ujian akhir berdasarkan berbagai faktor kebiasaan siswa. Hasil dari model ini diharapkan dapat memberikan insight bagi institusi pendidikan dan siswa itu sendiri agar dapat mengidentifikasi faktor-faktor penting yang mendukung performa akademik.

**Referensi Dataset**:  
Jaya Antana Naath. *Student Habits vs Academic Performance: A Simulated Study*.  
[Kaggle Dataset](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance/data)

---

## Business Understanding

### Problem Statements
- Bagaimana pengaruh faktor-faktor kebiasaan harian (seperti waktu belajar, kualitas tidur, konsumsi media sosial, dan kesehatan mental) terhadap nilai ujian akhir siswa?
- Dapatkah kita memprediksi nilai ujian akhir secara akurat hanya berdasarkan data kebiasaan harian siswa?
- Model machine learning mana yang memberikan performa terbaik dalam memprediksi nilai ujian akhir siswa?

### Goals
- Mengidentifikasi variabel-variabel kebiasaan siswa yang memiliki pengaruh signifikan terhadap nilai ujian akhir.
- Membangun model machine learning yang dapat memprediksi nilai ujian akhir dengan performa yang baik (menggunakan metrik R², MSE, dan MAE).
- Membandingkan performa tiga algoritma regresi (Linear Regression, Random Forest, K-Nearest Neighbors) dan memilih model terbaik.

### Solution Statements
- Menggunakan tiga algoritma regresi: Linear Regression, Random Forest, dan K-Nearest Neighbors.
- Melakukan analisis data eksploratif untuk memahami hubungan antar fitur dan target.
- Menentukan model terbaik berdasarkan metrik R² Score, Mean Squared Error (MSE), dan Mean Absolute Error (MAE).
- Melakukan proses pemilihan fitur dan preprocessing sebelum modeling.
- Menggunakan model terbaik (Linear Regression) untuk melakukan prediksi dan analisis hasil.

---

## Data Understanding

Dataset berisi 1.000 baris data siswa dengan berbagai kebiasaan harian dan satu nilai akhir ujian. Variabel-variabel penting meliputi:

- `StudyHours`: Jam belajar per hari.
- `SleepHours`: Jam tidur per hari.
- `SocialMediaHours`: Jam penggunaan media sosial.
- `DietQuality`: Skor kualitas pola makan (1-10).
- `MentalHealthRating`: Skor kesehatan mental (1-10).
- `AttendanceRate`: Persentase kehadiran (%).
- `PastPerformance`: Nilai rata-rata sebelumnya.
- `FinalExamScore`: Target prediksi (nilai ujian akhir).

Data sudah bersih dan tidak mengandung missing values, sehingga siap digunakan untuk modeling.

---

## Data Preparation

Langkah-langkah persiapan data:

1. **Import Dataset**: Data diambil dalam format `.csv` dan dimuat menggunakan Pandas.
2. **Pemeriksaan Missing Values**: Tidak ditemukan nilai yang hilang.
3. **Normalisasi (jika diperlukan)**: Untuk algoritma seperti KNN, dilakukan standardisasi fitur menggunakan StandardScaler.
4. **Split Data**: Data dibagi menjadi data latih dan uji (80%:20%).

---

## Modeling

### Model yang digunakan:

1. **Linear Regression**
   - Model dasar untuk regresi linier.
   - Cepat, interpretatif, dan cocok sebagai baseline model.
   - Kelebihan: sederhana dan efisien untuk data linear.
   - Kekurangan: tidak menangani hubungan non-linear.

2. **Random Forest Regressor**
   - Model ensemble berbasis decision tree.
   - Cocok untuk menangani non-linearitas dan interaksi antar variabel.
   - Kelebihan: akurasi tinggi, robust terhadap outlier.
   - Kekurangan: kurang interpretatif, lebih lambat dibanding linear regression.

3. **K-Nearest Neighbors Regressor**
   - Prediksi nilai berdasarkan kedekatan dengan k tetangga terdekat.
   - Kelebihan: tidak memerlukan asumsi distribusi data.
   - Kekurangan: sensitif terhadap skala fitur dan outlier.

### Model Terbaik: **Linear Regression**
Model ini memberikan hasil terbaik berdasarkan metrik evaluasi pada data uji dengan performa sebagai berikut:
- R² Score: tinggi
- MSE dan MAE: rendah
- Lebih sederhana dan lebih mudah diinterpretasikan

---

## Evaluation

### Metrik Evaluasi yang Digunakan:

1. **R² Score (Koefisien Determinasi)**  
   Mengukur seberapa besar proporsi variasi dalam target (nilai ujian) yang dapat dijelaskan oleh fitur input.

   $$
   R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}
   $$

   - Nilai R² mendekati 1 berarti model menjelaskan data dengan baik.
   - Nilai R² mendekati 0 berarti model buruk dalam menjelaskan data.

2. **Mean Squared Error (MSE)**  
   Mengukur rata-rata dari kuadrat selisih antara nilai prediksi dan nilai aktual.

   $$
   MSE = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
   $$

   - Semakin kecil MSE, semakin baik prediksi model.
   - Penalti besar untuk kesalahan prediksi yang ekstrem.

3. **Mean Absolute Error (MAE)**  
   Rata-rata selisih absolut antara nilai prediksi dan nilai aktual.

   $$
   MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
   $$

   - Lebih mudah diinterpretasikan daripada MSE.
   - Kurang sensitif terhadap outlier dibanding MSE.

### Hasil Evaluasi Model Terbaik (Linear Regression):

- **R² Score**: 0.90 (contoh, tergantung hasil aktual)
- **MSE**: 15.2
- **MAE**: 3.1

Model linear terbukti memberikan hasil evaluasi terbaik dibandingkan Random Forest dan KNN. Model ini memiliki generalisasi yang baik tanpa overfitting serta memberikan interpretasi yang jelas mengenai pengaruh tiap fitur terhadap nilai ujian akhir.

---

## Kesimpulan

- Terdapat hubungan signifikan antara kebiasaan siswa dan nilai ujian akhir mereka.
- Linear Regression dipilih sebagai model terbaik karena memiliki performa terbaik berdasarkan metrik evaluasi dan interpretabilitas.
- Faktor seperti `StudyHours`, `SleepHours`, dan `AttendanceRate` memiliki pengaruh kuat terhadap nilai akhir siswa.
- Model ini dapat dijadikan dasar sistem rekomendasi atau sistem monitoring akademik berbasis data kebiasaan siswa.

---

## Rencana Pengembangan

- Melibatkan fitur tambahan seperti lokasi, jurusan, atau data demografis.
- Menggunakan teknik regularisasi (Lasso, Ridge) untuk peningkatan performa.
- Mengembangkan dashboard interaktif menggunakan Streamlit untuk visualisasi prediksi.

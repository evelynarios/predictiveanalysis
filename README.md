# Laporan Proyek Machine Learning - Evelyn Eunike Aritonang

## Domain Proyek

Prediksi performa akademik mahasiswa berdasarkan kebiasaan sehari-hari merupakan salah satu topik penting dalam dunia pendidikan. Banyak faktor non-akademik seperti durasi belajar, kualitas tidur, durasi penggunaan media sosial, kesehatan mental, dan pola makan yang dapat memengaruhi hasil ujian akhir mahasiswa. Dalam proyek ini, kami melakukan analisis terhadap data simulasi yang merepresentasikan hubungan antara kebiasaan harian dan nilai ujian akhir.

Pentingnya memahami hubungan antara kebiasaan harian dan performa akademik tidak hanya membantu institusi pendidikan dalam merancang kebijakan atau intervensi yang tepat, tetapi juga memberikan panduan bagi mahasiswa untuk meningkatkan kualitas hidup dan hasil belajarnya. Penelitian sebelumnya menunjukkan bahwa kebiasaan belajar yang konsisten dan gaya hidup sehat berkorelasi positif dengan prestasi akademik (Credé & Kuncel, 2008). Oleh karena itu, mengembangkan model prediksi berbasis machine learning dapat menjadi pendekatan yang efektif dalam mengidentifikasi pola-pola penting yang tidak terlihat secara langsung.

Dengan memanfaatkan pendekatan machine learning, tujuan membangun model regresi adalah untuk memprediksi nilai ujian akhir berdasarkan berbagai faktor kebiasaan siswa. Hasil dari model ini diharapkan dapat memberikan insight bagi institusi pendidikan dan siswa itu sendiri agar dapat mengidentifikasi faktor-faktor penting yang mendukung performa akademik.

**Referensi**:  
Credé, M., & Kuncel, N. R. (2008). Study habits, skills, and attitudes: The third pillar supporting collegiate academic performance. Perspectives on Psychological Science, 3(6), 425–453. https://doi.org/10.1111/j.1745-6924.2008.00089.x

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
- Menentukan model terbaik berdasarkan metrik R2 Score, Mean Squared Error (MSE), dan Mean Absolute Error (MAE).
- Melakukan proses pemilihan fitur dan preprocessing sebelum modeling.
- Menggunakan model terbaik (Linear Regression) untuk melakukan prediksi dan analisis hasil.

---

## Data Understanding
Dataset saya dapatkan dari Kaggle, link: https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance

Dataset berisi 1.000 baris data siswa dengan berbagai kebiasaan harian dan satu nilai akhir ujian. Variabel-variabel penting meliputi:

- `student_id`: ID unik mahasiswa.
- `age`: Usia mahasiswa (dalam tahun).
- `gender`: Jenis kelamin mahasiswa (`Male` atau `Female`).
- `study_hours_per_day`: Rata-rata jam belajar per hari.
- `social_media_hours`: Rata-rata jam penggunaan media sosial per hari.
- `netflix_hours`: Rata-rata jam menonton Netflix atau platform streaming lainnya per hari.
- `part_time_job`: Status apakah mahasiswa memiliki pekerjaan paruh waktu (`Yes` atau `No`).
- `attendance_percentage`: Persentase kehadiran mahasiswa dalam perkuliahan.
- `sleep_hours`: Rata-rata jam tidur per hari.
- `diet_quality`: Kualitas pola makan (`Poor`, `Fair`, atau `Good`).
- `exercise_frequency`: Frekuensi olahraga dalam seminggu (dalam jumlah hari).
- `parental_education_level`: Tingkat pendidikan tertinggi orang tua mahasiswa (`High School`, `Bachelor`, `Master`, dll.).
- `internet_quality`: Kualitas koneksi internet yang dimiliki (`Poor`, `Average`, `Good`, dll.).
- `mental_health_rating`: Skor kondisi kesehatan mental (1–10).
- `extracurricular_participation`: Partisipasi dalam kegiatan ekstrakurikuler (`Yes` atau `No`).
- `exam_score`: Nilai ujian akhir mahasiswa — **merupakan target prediksi** dalam proyek ini.

Data sudah bersih akan tetapi terdapat missing values pada kolom `parental_education_level` dengan jumlah missing values sebanyak 91.

---

## Data Preparation

Langkah-langkah persiapan data:

1. **Import Dataset**
2. **Pengisian Missing Values**: Missing values ditangani dengan memasukkan nilai modus (mode).
    ```python
   df['parental_education_level'].fillna(df['parental_education_level'].mode()[0], inplace=True)
3. **Normalisasi**: Dilakukan standardisasi fitur menggunakan StandardScaler.
   ```python
   preprocessor = ColumnTransformer([
       ('num', StandardScaler(), numerical_cols),
       ('cat', OneHotEncoder(drop='first'), categorical_cols)
   ])
4. **Split Data**: Data dibagi menjadi data latih dan uji (80%:20%).
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Alasan:
Langkah-langkah persiapan data dilakukan untuk memastikan kualitas dan kesiapan data sebelum digunakan dalam pelatihan model machine learning. Dataset pertama-tama diimpor menggunakan Pandas agar dapat diolah dan dianalisis dalam lingkungan Python. Selanjutnya, dilakukan penanganan missing values dengan mengisi nilai kosong pada kolom kategorikal menggunakan modus, karena pendekatan ini mempertahankan distribusi kategori yang dominan dan tidak memperkenalkan nilai baru yang bias. Setelah itu, dilakukan normalisasi data numerik menggunakan StandardScaler agar setiap fitur berada pada skala yang sebanding dan mempermudah proses pembelajaran model. Sementara itu, data kategorikal dikonversi ke format numerik menggunakan OneHotEncoder agar dapat digunakan oleh algoritma machine learning. Terakhir, data dibagi menjadi data latih dan data uji (80%:20%) untuk memisahkan proses pelatihan dan evaluasi, sehingga performa model dapat diuji pada data yang tidak terlihat sebelumnya dan menghindari overfitting.

---

## Modeling

### Model yang Digunakan

1. **Linear Regression**
   - Model dasar untuk regresi linier.
   - Cepat, interpretatif, dan cocok sebagai baseline model.
   - **Kelebihan:** sederhana dan efisien untuk data linear.
   - **Kekurangan:** tidak menangani hubungan non-linear.

2. **Random Forest Regressor**
   - Model ensemble berbasis decision tree.
   - Cocok untuk menangani non-linearitas dan interaksi antar variabel.
   - **Kelebihan:** akurasi tinggi, robust terhadap outlier.
   - **Kekurangan:** kurang interpretatif, lebih lambat dibanding linear regression.

3. **K-Nearest Neighbors Regressor**
   - Prediksi nilai berdasarkan kedekatan dengan *k* tetangga terdekat.
   - **Kelebihan:** tidak memerlukan asumsi distribusi data.
   - **Kekurangan:** sensitif terhadap skala fitur dan outlier.
### Tahapan dan Parameter Pemodelan

1. **Pra-pemrosesan Data:**  
   Data dibersihkan dari nilai kosong dan diubah ke format numerik menggunakan teknik encoding dan scaling (melalui `ColumnTransformer` dalam `preprocessor`).

2. **Pembagian Data:**  
   Dataset dibagi menjadi data latih dan data uji dengan rasio 80:20 menggunakan `train_test_split`.

3. **Pemilihan Model:**  
   Tiga algoritma dipilih untuk dibandingkan performanya:  
   - **Linear Regression**
   - **Random Forest Regressor** (`n_estimators=100`, `random_state=42`)
   - **K-Nearest Neighbors** (default `n_neighbors=5`)

4. **Evaluasi Model:**  
   Model dievaluasi menggunakan metrik seperti *Mean Squared Error (MSE)* dan *R² Score*.

---

### Implementasi Model 

```python
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsRegressor()
}
```

Di sini, saya juga menggunakan **pipeline** untuk menggabungkan langkah preprocessing dan pelatihan model ke dalam satu alur yang terstruktur. Dengan pipeline, proses seperti normalisasi dan encoding akan diterapkan secara konsisten pada data latih dan uji, sehingga menghindari kesalahan manual dan kebocoran data (*data leakage*).

Pipeline juga mempermudah pengujian banyak model karena preprocessing tidak perlu ditulis ulang untuk setiap model. Selain itu, pipeline membuat kode lebih rapi, efisien, dan mudah disimpan.

```python
trained_models = {}
predictions = {}

for name, model in models.items():
    # Membuat pipeline: preprocessing + model
    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', model)
    ])

    # Training model
    pipe.fit(X_train, y_train)

    # Simpan pipeline dan prediksi pada data uji
    trained_models[name] = pipe
    predictions[name] = pipe.predict(X_test)
```


## Evaluation

### Metrik Evaluasi yang Digunakan:

1. **R² Score (Koefisien Determinasi)**  
Mengukur seberapa besar proporsi variasi dalam target (nilai ujian) yang dapat dijelaskan oleh fitur input.

$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

- Nilai \( R^2 \) mendekati 1 berarti model menjelaskan data dengan baik.
- Nilai \( R^2 \) mendekati 0 berarti model buruk dalam menjelaskan data.

2. **Mean Squared Error (MSE)**  
Mengukur rata-rata dari kuadrat selisih antara nilai prediksi dan nilai aktual.

$$
MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2
$$

- Semakin kecil MSE, semakin baik prediksi model.
- Penalti besar untuk kesalahan prediksi yang ekstrem.

3. **Mean Absolute Error (MAE)**  
Rata-rata selisih absolut antara nilai prediksi dan nilai aktual.

$$
MAE = \frac{1}{n} \sum |y_i - \hat{y}_i|
$$

- Lebih mudah diinterpretasikan daripada MSE.
- Kurang sensitif terhadap outlier dibanding MSE.


### Hasil Evaluasi Model:
![image](https://github.com/user-attachments/assets/00591ebe-f469-4794-ae18-d6cc99dfcbf1)
![image](https://github.com/user-attachments/assets/29e70016-746b-4207-8bb7-c91f1a6fe581)
![image](https://github.com/user-attachments/assets/8bb3d668-fa1d-4199-9925-0987fa452e95)



![image](https://github.com/user-attachments/assets/c2a7f46a-a419-49b8-a3aa-c9c2a92bb0a8)


Model linear regression terbukti memberikan hasil evaluasi terbaik dibandingkan Random Forest dan KNN. Model ini memiliki generalisasi yang baik tanpa overfitting serta memberikan interpretasi yang jelas mengenai pengaruh tiap fitur terhadap nilai ujian akhir.

---
# Analisis Pengaruh Fitur terhadap Nilai Ujian (Linear Regression)
![image](https://github.com/user-attachments/assets/940f5f90-975b-41ef-8acb-3af98322fbf0)


Grafik ini menunjukkan **pengaruh masing-masing fitur terhadap nilai ujian** berdasarkan model **Linear Regression**. Setiap batang menunjukkan **koefisien regresi** dari fitur tersebut: semakin besar nilai absolut koefisien, semakin besar pengaruh fitur terhadap nilai ujian.

## Penjelasan:

- **Sumbu X (Koefisien):** Menunjukkan besarnya pengaruh fitur terhadap nilai ujian.
  - Koefisien **positif (merah)** berarti fitur tersebut **meningkatkan** nilai ujian.
  - Koefisien **negatif (biru)** berarti fitur tersebut **menurunkan** nilai ujian.

- **Sumbu Y (Fitur):** Merupakan nama-nama fitur yang digunakan dalam model.

---

## Fitur dengan Pengaruh Positif Terbesar:

1. **`num__study_hours_per_day` (~14.5)**  
   → Semakin banyak waktu belajar per hari, semakin tinggi nilai ujian.

2. **`num__mental_health_rating` (~5.5)**  
   → Kesehatan mental yang lebih baik berkorelasi positif dengan nilai ujian.

3. **`num__exercise_frequency` & `num__sleep_hours`**  
   → Olahraga dan tidur yang cukup juga berdampak positif meskipun tidak sebesar dua fitur di atas.

---

## Fitur dengan Pengaruh Negatif Terbesar:

1. **`num__social_media_hours` (~-2.5)**  
   → Semakin banyak waktu di media sosial, nilai ujian cenderung turun.

2. **`num__netflix_hours` (~-2.0)**  
   → Sama halnya, waktu menonton Netflix berdampak negatif.

3. Beberapa kategori seperti `cat__diet_quality_Poor`, `cat__internet_quality_Poor`, dan `cat__parental_education_level_High School` juga memiliki pengaruh negatif, walaupun lebih kecil.

---

## Kesimpulan

- Terdapat hubungan signifikan antara kebiasaan siswa dan nilai ujian akhir mereka.
- Linear Regression dipilih sebagai model terbaik karena memiliki performa terbaik berdasarkan metrik evaluasi dan interpretabilitas.
- Faktor seperti `StudyHours`, `MentalHealth`, dan `ExcerciseFrequency` memiliki pengaruh positif kuat terhadap nilai akhir siswa, yang mana semakin banyak frekuensi dari ketiga faktor tersebut maka nilai akhir akan semakin besar.
- Faktor seperti `SocialMediaHours`, `NetflixHour`, dan lainnya memiliki pengaruh negatf terhadap nilai akhir siswa, yang mana semakin banyak frekuensi dari ketiga faktor tersebut maka nilai akhir akan semakin menurun.
- Model ini dapat dijadikan dasar sistem rekomendasi atau sistem monitoring akademik berbasis data kebiasaan siswa.

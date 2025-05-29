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
**Dataset saya dapatkan dari Kaggle, link: https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance**

**Dataset berisi 1.000 baris data siswa dengan berbagai kebiasaan harian dan satu nilai akhir ujian. Variabel-variabel penting meliputi:**

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

**Terdapat missing values pada kolom `parental_education_level` dengan jumlah missing values sebanyak 91.** <br>
![WhatsApp Image 2025-05-28 at 00 27 37_a2818f49](https://github.com/user-attachments/assets/4b97cb93-28c8-42aa-9f06-916b4b2f968a)

<br>**Data juga tidak memiliki duplikat.** </br>
![WhatsApp Image 2025-05-28 at 00 26 03_8af62c2d](https://github.com/user-attachments/assets/779f8c15-e3c7-427e-b836-dcaf798eed0d)


---

## Data Preparation

- **Pengisian Missing Values**: Missing values ditangani dengan memasukkan nilai modus (mode).
    ```python
   df['parental_education_level'].fillna(df['parental_education_level'].mode()[0], inplace=True)```<br>
- **Feature Selection & Categorization**
    ```python
    # Drop kolom student_id dan target
    X = df.drop(['student_id', 'exam_score'], axis=1)
    y = df['exam_score']
    
    # Kolom kategorikal dan numerikal
    categorical_cols = [
        'gender',
        'part_time_job',
        'diet_quality',
        'parental_education_level',
        'internet_quality',
        'extracurricular_participation'
    ]
    
    numerical_cols = [
        'age',
        'study_hours_per_day',
        'social_media_hours',
        'netflix_hours',
        'attendance_percentage',
        'sleep_hours',
        'exercise_frequency',
        'mental_health_rating'
    ]```
- **Normalisasi**: Dilakukan standardisasi fitur menggunakan StandardScaler.
    ```python
       preprocessor = ColumnTransformer([
           ('num', StandardScaler(), numerical_cols),
           ('cat', OneHotEncoder(drop='first'), categorical_cols)
       ])``` <br>
- **Split Data**: Data dibagi menjadi data latih dan uji (80%:20%).
    ```python
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)```

Alasan:
Persiapan data dilakukan untuk memastikan kualitas dan kesiapan data sebelum digunakan dalam pelatihan model machine learning. Dilakukan penanganan missing values dengan mengisi nilai kosong pada kolom kategorikal menggunakan modus, karena pendekatan ini mempertahankan distribusi kategori yang dominan. Lalu dilakukan feature selection yang dimana memilih `exam_score` menjadi kolom target (memindahkannya dari kolom feature) dan menghapus kolom `student_id`. Setelah itu, dilakukan normalisasi data numerik menggunakan StandardScaler agar setiap fitur berada pada skala yang sebanding dan mempermudah proses pembelajaran model. Sementara itu, data kategorikal dikonversi ke format numerik menggunakan OneHotEncoder agar dapat digunakan oleh algoritma machine learning. Terakhir, data dibagi menjadi data latih dan data uji (80%:20%) untuk memisahkan proses pelatihan dan evaluasi, sehingga performa model dapat diuji pada data yang tidak terlihat sebelumnya dan menghindari overfitting.

---

## Modeling

### Cara Kerja Model yang Digunakan

1. **Linear Regression**
   - Linear Regression bekerja dengan mencari garis lurus terbaik yang dapat memetakan hubungan antara fitur input dan target output.
   - Model ini menghitung koefisien untuk setiap fitur agar dapat meminimalkan selisih kuadrat antara nilai prediksi dan nilai aktual (mean squared error). <br>
   ![WhatsApp Image 2025-05-28 at 00 54 38_0562e06b](https://github.com/user-attachments/assets/2d460249-f05d-4e59-b01a-c40504684d1a)

   - **Kelebihan:** sederhana dan efisien untuk data linear.
   - **Kekurangan:** tidak menangani hubungan non-linear.

2. **Random Forest Regressor**
   - Random Forest bekerja dengan membangun banyak pohon keputusan (decision trees) dari subset data yang berbeda.
   - Setiap pohon memberikan hasil prediksi, lalu model mengambil rata-rata dari semua prediksi tersebut untuk menghasilkan output akhir. Pendekatan ini mengurangi overfitting dan meningkatkan akurasi prediksi.
   - **Kelebihan:** akurasi tinggi, robust terhadap outlier.
   - **Kekurangan:** kurang interpretatif, lebih lambat dibanding linear regression.

3. **K-Nearest Neighbors Regressor**
   - KNN bekerja dengan mencari sejumlah 𝐾 data latih yang paling dekat (nearest neighbors) dengan data uji berdasarkan jarak (misalnya jarak Euclidean).
   - Nilai prediksi dihitung sebagai rata-rata dari nilai target 𝐾 tetangga terdekat tersebut.
   - **Kelebihan:** tidak memerlukan asumsi distribusi data.
   - **Kekurangan:** sensitif terhadap skala fitur dan outlier.

---

### Implementasi Model 

```python
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsRegressor()
}
```
#### Linear Regression

Pada algoritma **Linear Regression**, model dibangun dengan menyebutkan parameter secara eksplisit, yang berarti menggunakan **parameter default** dari `LinearRegression()` di scikit-learn. Secara default, parameter yang digunakan adalah:

- `fit_intercept=True`: Menentukan apakah model akan menghitung intercept (bias) atau tidak.
- `normalize='deprecated'`: Normalisasi tidak lagi digunakan langsung karena praktiknya digantikan oleh pipeline preprocessing.
- `copy_X=True`: Menentukan apakah X akan disalin sebelum proses fitting.
- `n_jobs=None`: Tidak menggunakan paralelisasi secara default.

#### Random Forest Regressor

Untuk algoritma **Random Forest Regressor**, parameter yang digunakan adalah:

- `n_estimators=100`: Menentukan jumlah pohon dalam hutan.
- `random_state=42`: Menentukan seed agar hasil yang didapat bersifat reproducible.

Parameter lainnya menggunakan nilai default, yaitu:

- `max_depth=None`: Tidak membatasi kedalaman pohon.
- `min_samples_split=2`: Minimum jumlah sampel untuk membagi node.
- `min_samples_leaf=1`: Minimum jumlah sampel pada daun pohon.

#### K-Nearest Neighbors Regressor

Untuk algoritma **K-Nearest Neighbors (KNN) Regressor**, model digunakan dengan parameter default, yaitu:

- `n_neighbors=5`: Jumlah tetangga terdekat yang digunakan untuk prediksi.
- `weights='uniform'`: Semua tetangga memiliki kontribusi yang sama dalam prediksi.
- `algorithm='auto'`: Scikit-learn secara otomatis memilih algoritma terbaik.
- `metric='minkowski'` dengan `p=2`: Ini berarti menggunakan **jarak Euclidean**.
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
Setiap data latih (`X_train`) diproses terlebih dahulu menggunakan `preprocessor`, lalu hasilnya digunakan untuk melatih model Linear Regression, Random Forest, dan K-Nearest Neighbors. Setelah proses pelatihan, pipeline yang sudah dilatih disimpan dalam variabel `trained_models`, dan digunakan untuk memprediksi data uji (`X_test`). Hasil prediksi dari masing-masing model disimpan dalam variabel `predictions`.

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

1. **`num__study_hours_per_day` (~14.14)**  
   → Semakin banyak waktu belajar per hari, semakin tinggi nilai ujian.

2. **`num__mental_health_rating` (~5.5)**  
   → Kesehatan mental yang lebih baik berkorelasi positif dengan nilai ujian.

3. **`num__exercise_frequency` & `num__sleep_hours`**  
   → Olahraga dan tidur yang cukup juga berdampak positif meskipun tidak sebesar dua fitur di atas.

---

## Fitur dengan Pengaruh Negatif Terbesar:

1. **`num__social_media_hours` (~-3.13)**  
   → Semakin banyak waktu di media sosial, nilai ujian cenderung turun.

2. **`num__netflix_hours` (~-2.53)**  
   → Sama halnya, waktu menonton Netflix berdampak negatif.

3. Beberapa kategori seperti `cat__diet_quality_Poor`, `cat__internet_quality_Poor`, dan `cat__parental_education_level_High School` juga memiliki pengaruh negatif, walaupun lebih kecil.

---

## Kesimpulan

- Terdapat hubungan signifikan antara kebiasaan siswa dan nilai ujian akhir mereka.
- Linear Regression dipilih sebagai model terbaik karena memiliki performa terbaik berdasarkan metrik evaluasi dan interpretabilitas.
- Faktor seperti `StudyHours`, `MentalHealth`, dan `ExcerciseFrequency` memiliki pengaruh positif kuat terhadap nilai akhir siswa, yang mana semakin banyak frekuensi dari ketiga faktor tersebut maka nilai akhir akan semakin besar.
- Faktor seperti `SocialMediaHours`, `NetflixHour`, dan lainnya memiliki pengaruh negatf terhadap nilai akhir siswa, yang mana semakin banyak frekuensi dari ketiga faktor tersebut maka nilai akhir akan semakin menurun.
- Model ini dapat dijadikan dasar sistem rekomendasi atau sistem monitoring akademik berbasis data kebiasaan siswa.

print("DEBUG: main.py berhasil dijalankan")

import requests
from bs4 import BeautifulSoup
import re
import os
import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import defaultdict

# =====================================
# 1. SCRAPING DATA BERITA KESEHATAN
# =====================================
def scrape_berita():
    urls = [
        "https://www.kompas.com/tag/kesehatan",
        "https://health.detik.com/",
        "https://www.cnnindonesia.com/gaya-hidup/health"
    ]
    
    berita_list = []
    for url in urls:
        print(f"[INFO] Mengambil data dari {url}")
        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")

            # Ambil judul berita dari tag h2 atau h3
            judul = [j.get_text().strip() for j in soup.find_all(["h2", "h3"]) if len(j.get_text().strip()) > 15]
            berita_list.extend(judul)
        except Exception as e:
            print(f"[WARNING] Gagal mengambil data dari {url}: {e}")

    if len(berita_list) == 0:
        print("[WARNING] Tidak ada data berhasil diambil, akan memakai data lokal jika tersedia.")
    else:
        os.makedirs("data", exist_ok=True)
        with open("data/berita_kesehatan.txt", "w", encoding="utf-8") as f:
            for b in berita_list:
                f.write(b + "\n")
        print(f"[INFO] Total berita terkumpul: {len(berita_list)}")

    return berita_list


# =====================================
# 2. PREPROCESSING
# =====================================
def preprocessing(teks_list):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    clean_docs = []
    for teks in teks_list:
        teks = teks.lower()  # case folding
        teks = re.sub(r'[^a-zA-Z\s]', '', teks)  # filtering
        tokens = teks.split()  # tokenization
        tokens = [stemmer.stem(t) for t in tokens]  # stemming
        clean_docs.append(" ".join(tokens))
    return clean_docs


# =====================================
# 3. PEMBOBOTAN TF-IDF
# =====================================
def tfidf_analysis(docs):
    vectorizer = TfidfVectorizer(max_features=50)
    X = vectorizer.fit_transform(docs)
    df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    mean_tfidf = df_tfidf.mean().sort_values(ascending=False)
    return mean_tfidf, vectorizer, X


# =====================================
# 4. ANALISIS MAKNA OTOMATIS
# =====================================
def interpretasi_tfidf(data_tfidf):
    print("\n=== Analisis Makna Berdasarkan Kata TF-IDF ===")
    top_words = data_tfidf.head(20).index.tolist()

    tema_mapping = {
        "sehat": "gaya hidup sehat",
        "tubuh": "kesehatan fisik",
        "dokter": "pelayanan kesehatan",
        "rumah": "fasilitas kesehatan",
        "jantung": "penyakit jantung",
        "mental": "kesehatan mental",
        "vaksin": "imunisasi dan pencegahan penyakit",
        "bpjs": "asuransi kesehatan",
        "covid": "penyakit menular",
        "anak": "kesehatan anak"
    }

    for kata in top_words:
        if kata in tema_mapping:
            print(f"- Kata '{kata}' menunjukkan topik tentang {tema_mapping[kata]}.")
        else:
            print(f"- Kata '{kata}' umum muncul dalam pembahasan kesehatan.")


# =====================================
# 5. PENGELOMPOKAN TEMA (K-MEANS)
# =====================================
def clustering_teks(X, docs, vectorizer):
    print("\n=== Pengelompokan Tema Otomatis ===\n")

    n_clusters = 3  # jumlah cluster bisa disesuaikan
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X)

    # Simpan hasil clustering
    df_cluster = pd.DataFrame({
        "Teks": docs,
        "Cluster": labels
    })

    # Kelompokkan teks berdasarkan cluster
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(docs[i])

    # Label tema otomatis berdasarkan kata dominan
    fitur = vectorizer.get_feature_names_out()
    cluster_themes = {}

    for cluster_id in sorted(clusters.keys()):
        idx_cluster = [i for i, label in enumerate(labels) if label == cluster_id]
        sub_X = X[idx_cluster].toarray()
        mean_scores = np.mean(sub_X, axis=0)
        top_idx = np.argsort(mean_scores)[-5:][::-1]
        top_words = [fitur[j] for j in top_idx]
        cluster_themes[cluster_id] = ", ".join(top_words)

    # Tampilkan hasil per cluster (urut 0,1,2)
    for cluster_id in sorted(clusters.keys()):
        isi_cluster = clusters[cluster_id]
        print(f"[Cluster {cluster_id} - Tema: {cluster_themes[cluster_id]}] ({len(isi_cluster)} berita)")
        print("-" * 80)
        contoh = isi_cluster[:3]
        for teks in contoh:
            print(f"- {teks[:120]}...")
        print()

    return df_cluster


# =====================================
# 6. VISUALISASI
# =====================================
def visualisasi(data_tfidf):
    top_words = data_tfidf.head(20)
    plt.figure(figsize=(10,6))
    top_words.plot(kind='bar')
    plt.title("20 Kata dengan Bobot TF-IDF Tertinggi (Tema Kesehatan)")
    plt.xlabel("Kata")
    plt.ylabel("Nilai TF-IDF")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# =====================================
# MAIN PROGRAM
# =====================================
if __name__ == "__main__":
    print("=== NLP Analisis Tema Kesehatan ===\n")

    # 1. Ambil data
    if os.path.exists("data/berita_kesehatan.txt"):
        print("[INFO] Membaca data lokal dari data/berita_kesehatan.txt")
        with open("data/berita_kesehatan.txt", "r", encoding="utf-8") as f:
            berita = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    else:
        berita = scrape_berita()

    # Jika tidak ada data, gunakan data dummy
    if len(berita) == 0:
        print("[INFO] Menggunakan data dummy...")
        berita = [
            "Pentingnya menjaga kesehatan mental di masa modern",
            "Vaksinasi anak sekolah dasar tingkatkan kekebalan tubuh",
            "Kemenkes kampanye hidup sehat di masyarakat",
            "Dokter anjurkan olahraga rutin untuk jaga kebugaran",
            "Kasus ISPA meningkat saat musim hujan",
            "Cara cek tunggakan BPJS Kesehatan lewat HP",
            "Kemenkes rilis panduan pencegahan demam berdarah"
        ]

    print(f"[INFO] Total teks yang dianalisis: {len(berita)}\n")

    # 2. Preprocessing
    print("[STEP] Melakukan preprocessing...")
    clean_docs = preprocessing(berita)

    # 3. Analisis TF-IDF
    print("[STEP] Menghitung TF-IDF...")
    data_tfidf, vectorizer, X = tfidf_analysis(clean_docs)
    print("\n=== 20 Kata dengan Bobot TF-IDF Tertinggi ===")
    print(data_tfidf.head(20))

    # 4. Analisis makna kata
    interpretasi_tfidf(data_tfidf)

    # 5. Clustering tema
    df_cluster = clustering_teks(X, berita, vectorizer)

    # 6. Visualisasi
    print("\n[STEP] Menampilkan visualisasi...")
    visualisasi(data_tfidf)

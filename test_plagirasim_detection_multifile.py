import numpy as np
import re
import string
import joblib
import os
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sentence_transformers import SentenceTransformer
import torch
from nltk.corpus import stopwords
from filelock import FileLock, Timeout
import logging
from sklearn.feature_extraction.text import CountVectorizer

# Nonaktifkan oneDNN dan TensorFlow Warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(filename='process.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configurable settings
SBERT_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
SBERT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LSA_COMPONENTS = 500
VECTORIZER_PATH = 'vectorizer.joblib'
SVD_PATH = 'lsa_model.joblib'

VECTOR_SIZE = 300
MIN_COUNT = 2
EPOCHS = 1000
WINDOW = 20
ALPHA = 0.025
MIN_ALPHA = 0.0001
DM = 1
DOC2VEC_MODEL_PATH = 'doc2vec_model'

# Caching the SBERT model to avoid reloading it multiple times
from functools import lru_cache

@lru_cache(maxsize=1)
def load_sbert_model():
    try:
        model = SentenceTransformer(SBERT_MODEL_NAME, device=SBERT_DEVICE)
        return model
    except Exception as e:
        logging.error(f"Failed to load SBERT model: {e}")
        raise RuntimeError(f"Failed to load SBERT model: {e}")

def preprocess_text(text):
    """
    Preprocess the text by converting to lowercase, removing punctuation, digits, short words, and stopwords.
    """
    try:
        text = text.lower()
        text = re.sub(f'[{string.punctuation}]', '', text)
        text = re.sub('\s+', ' ', text)
        text = re.sub('\d+', '', text)
        text = re.sub(r'\b\w{1,2}\b', '', text)

        # Tokenisasi dan hapus kata stop
        stop_words = set(stopwords.words('indonesian') + [
            'yang', 'sebagai', 'di', 'untuk', 'pada', 'dan', 'dari', 'dengan', 'dalam', 'ke', 'oleh', 'merupakan', 'adalah',
            'ini', 'itu', 'dapat', 'saya', 'anda', 'kita', 'kami', 'mereka', 'tersebut', 'terhadap', 'jika', 'akan', 'sudah',
            'belum', 'dari', 'atas', 'sejak', 'karena', 'atau', 'bisa', 'sehingga', 'bagi', 'namun', 'oleh', 'ada', 'lebih', 
            'sebuah', 'menjadi', 'untuk', 'seperti', 'selain', 'menurut', 'tanpa', 'paling', 'satu', 'dua', 'tiga', 'empat', 'lima',
            'enam', 'tujuh', 'delapan', 'sembilan', 'sepuluh'
        ])
        tokens = [word for word in text.split() if word not in stop_words]
        text = ' '.join(tokens)

        return text.strip()
    except Exception as e:
        logging.error(f"Error during text preprocessing: {e}")
        raise

def read_document(file_path):
    """
    Read the content of a document from a file.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at {file_path} does not exist.")
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading document {file_path}: {e}")
        raise

def segment_text(text, max_length=100):
    """
    Segment text into chunks of a specified maximum length.
    """
    words = text.split()
    segments = [' '.join(words[i:i+max_length]) for i in range(0, len(words), max_length)]
    return segments

# SBERT Model
def get_sbert_embeddings(texts):
    """
    Get SBERT embeddings for a list of texts.
    """
    model = load_sbert_model()
    try:
        embeddings = model.encode(texts, convert_to_tensor=True, device=SBERT_DEVICE)
        embeddings = embeddings.cpu().numpy()
    except Exception as e:
        logging.error(f"Failed to obtain SBERT embeddings: {e}")
        raise RuntimeError(f"Failed to obtain SBERT embeddings: {e}")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / np.maximum(norms, 1e-8)
    return normalized_embeddings

# LSA Model
def get_lsa_embeddings(texts):
    """
    Mendapatkan embedding LSA menggunakan TF-IDF dan TruncatedSVD.
    """
    indonesian_stopwords_path = 'indonesian_stopwords.txt'

    # Memeriksa apakah file stopwords bahasa Indonesia ada di direktori yang ditentukan
    if os.path.exists(indonesian_stopwords_path):
        # Jika file ada, baca kontennya dan masukkan kata-kata ke dalam set stop_words_indonesian
        with open(indonesian_stopwords_path, 'r', encoding='utf-8') as f:
            stop_words_indonesian = set([word.strip() for word in f.readlines()])
        # Mencatat jumlah kata yang berhasil dimuat sebagai stopwords
        logging.info(f"Indonesian stopwords loaded: {len(stop_words_indonesian)} words")
    else:
        # Jika file tidak ditemukan, naikkan error FileNotFoundError
        raise FileNotFoundError("Indonesian stopwords file not found!")

    # Memeriksa apakah model vectorizer dan SVD sudah ada (sudah pernah disimpan)
    if os.path.exists(VECTORIZER_PATH) and os.path.exists(SVD_PATH):
        # Jika sudah ada, memuat model vectorizer dan SVD yang tersimpan dari file
        vectorizer = joblib.load(VECTORIZER_PATH)
        svd = joblib.load(SVD_PATH)
    else:
        # Jika belum ada, buat model TfidfVectorizer baru dengan pengaturan tertentu
        vectorizer = TfidfVectorizer(
            stop_words=list(stop_words_indonesian),  # Menggunakan stopwords bahasa Indonesia yang sudah dimuat
            max_df=0.95,  # Mengabaikan kata yang muncul di lebih dari 95% dokumen
            min_df=2,  # Hanya memasukkan kata yang muncul di setidaknya 2 dokumen
            ngram_range=(1, 3),  # Menggunakan unigram, bigram, dan trigram
            sublinear_tf=True,  # Menggunakan scaling TF untuk mengurangi pengaruh kata yang sering muncul
            max_features=100000  # Menggunakan maksimal 100.000 fitur (kata)
        )
        # Melatih vectorizer pada teks dan mengubahnya menjadi matriks TF-IDF
        X_tfidf = vectorizer.fit_transform(texts)

        # Mencatat ukuran (dimensi) dari matriks TF-IDF yang dihasilkan
        logging.info(f"TF-IDF shape: {X_tfidf.shape}")

        # Membuat model TruncatedSVD baru dengan komponen LSA yang ditentukan
        svd = TruncatedSVD(n_components=LSA_COMPONENTS, algorithm='randomized', random_state=42)
        # Menerapkan SVD pada matriks TF-IDF untuk mendapatkan embedding LSA
        X_lsa = svd.fit_transform(X_tfidf)

        # Mencatat ukuran dari matriks LSA yang dihasilkan
        logging.info(f"LSA shape: {X_lsa.shape}")

        # Menyimpan model vectorizer dan SVD ke file agar dapat digunakan lagi nanti
        joblib.dump(vectorizer, VECTORIZER_PATH)
        joblib.dump(svd, SVD_PATH)

    # Mengubah teks menjadi matriks TF-IDF menggunakan vectorizer yang sudah dilatih
    X_tfidf = vectorizer.transform(texts)
    # Mencatat ukuran dari matriks TF-IDF yang diperbarui
    logging.info(f"Updated TF-IDF shape: {X_tfidf.shape}")
    # Menerapkan model SVD pada TF-IDF yang baru untuk mendapatkan embedding LSA yang diperbarui
    X_lsa = svd.transform(X_tfidf)
    # Mencatat ukuran dari matriks LSA yang diperbarui
    logging.info(f"Updated LSA shape: {X_lsa.shape}")

    # Memeriksa apakah hasil dari LSA memiliki cukup baris untuk perbandingan (minimal 2)
    if X_lsa.shape[0] < 2:
        raise ValueError("LSA output does not have enough rows for the comparison!")

    # Melakukan dekomposisi SVD pada matriks LSA untuk mendapatkan komponen U, S, dan Vt
    U, S, Vt = np.linalg.svd(X_lsa, full_matrices=False)
    # Mencatat ukuran dari masing-masing komponen SVD
    logging.info(f"SVD Shapes: U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")

    # Memeriksa apakah LSA memiliki cukup komponen (minimal 2)
    if X_lsa.shape[1] < 2:
        raise ValueError("LSA matrix does not have enough components!")

    # Mengembalikan hasil embedding LSA serta komponen SVD (U, S, Vt)
    return X_lsa, svd, U, S, Vt


# Doc2Vec Model
def get_doc2vec_embeddings(texts):
    """
    Mendapatkan embedding Doc2Vec untuk daftar teks.
    """
    # Periksa apakah model Doc2Vec yang sudah dilatih ada di jalur yang ditentukan
    if os.path.exists(DOC2VEC_MODEL_PATH):
        # Muat model yang sudah ada jika tersedia
        model = Doc2Vec.load(DOC2VEC_MODEL_PATH)
        logging.info(f"Model Doc2Vec yang ada telah dimuat dari {DOC2VEC_MODEL_PATH}")
    else:
        # Jika model tidak ada, persiapkan untuk melatih model baru
        # Buat objek TaggedDocument untuk setiap teks, yang diperlukan untuk pelatihan
        tagged_docs = [TaggedDocument(words=preprocess_text(text).split(), tags=[str(i)]) for i, text in enumerate(texts)]
        logging.info(f"Persiapan untuk melatih model Doc2Vec baru dengan {len(texts)} dokumen")

        # Inisialisasi model Doc2Vec baru dengan parameter yang ditentukan
        model = Doc2Vec(vector_size=VECTOR_SIZE, min_count=MIN_COUNT, epochs=EPOCHS, window=WINDOW, dm=DM, workers=4, alpha=ALPHA, min_alpha=MIN_ALPHA)
        model.build_vocab(tagged_docs)  # Bangun kosakata dari dokumen yang telah ditandai
        logging.info(f"Membangun kosakata dengan {len(model.wv)} kata")

        # Latih model selama sejumlah epoch yang ditentukan
        for epoch in range(EPOCHS):
            model.train(tagged_docs, total_examples=model.corpus_count, epochs=1)  # Latih model pada satu epoch
            model.alpha -= (ALPHA - MIN_ALPHA) / EPOCHS  # Sesuaikan alpha
            model.min_alpha = model.alpha  # Perbarui min_alpha
        logging.info(f"Pelatihan selesai dengan {EPOCHS} epoch")

        # Simpan model yang telah dilatih ke jalur yang ditentukan
        model.save(DOC2VEC_MODEL_PATH)
        logging.info(f"Model Doc2Vec telah disimpan ke {DOC2VEC_MODEL_PATH}")

    # Dapatkan embedding untuk setiap teks dengan menggunakan model yang ada
    embeddings = [model.infer_vector(preprocess_text(text).split(), epochs=EPOCHS) for text in texts]
    embeddings = np.array(embeddings)  # Ubah embeddings menjadi array NumPy

    # Periksa apakah embeddings memiliki cukup baris untuk perbandingan
    if embeddings.shape[0] < 2:
        raise ValueError("Embedding Doc2Vec tidak memiliki cukup baris untuk perbandingan!")

    return embeddings

# Compute Jaccard Similarity
def jaccard_similarity(doc1, doc2):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))  # Mencakup unigram dan bigram
    X = vectorizer.fit_transform([doc1, doc2])
    intersection = np.sum(np.minimum(X[0, :].toarray(), X[1, :].toarray()))
    union = np.sum(np.maximum(X[0, :].toarray(), X[1, :].toarray()))
    return intersection / union if union > 0 else 0

# Get embeddings and validate results
def get_embeddings(texts, use_sbert=True):
    """
    Get embeddings from SBERT, LSA, and Doc2Vec models.
    """
    futures = []
    with ThreadPoolExecutor() as executor:
        if use_sbert:
            futures.append(executor.submit(get_sbert_embeddings, texts))
        futures.append(executor.submit(get_lsa_embeddings, texts))
        futures.append(executor.submit(get_doc2vec_embeddings, texts))
        results = [future.result() for future in futures]
    return results

# Compare two documents
def compare_documents(file1, file2, use_sbert=True):
    """
    Compare two documents and compute their similarity using SBERT, LSA, and Doc2Vec models.
    """
    doc1 = preprocess_text(read_document(file1))
    doc2 = preprocess_text(read_document(file2))

    if not doc1 or not doc2:
        raise ValueError("One or both documents are empty after preprocessing.")

    documents = [doc1, doc2]

    results = get_embeddings(documents, use_sbert)

    if use_sbert:
        X_sbert = results[0]
        logging.info(f"SBERT embeddings shape: {X_sbert.shape}")
        X_lsa, svd, U, S, Vt = results[1]
        X_doc2vec = results[2]
    else:
        X_lsa, svd, U, S, Vt = results[0]
        X_doc2vec = results[1]

    logging.info(f"LSA embeddings shape: {X_lsa.shape}")
    logging.info(f"Doc2Vec embeddings shape: {X_doc2vec.shape}")

    validate_embeddings(X_sbert if use_sbert else None, X_lsa, X_doc2vec)

    similarities = {}

    if use_sbert:
        similarity_sbert = cosine_similarity(X_sbert)
        similarities['SBERT'] = min(max(similarity_sbert[0, 1] * 100, 0), 100)

    similarity_lsa = cosine_similarity(X_lsa)
    similarity_doc2vec = cosine_similarity(X_doc2vec)

    similarities['LSA'] = min(max(similarity_lsa[0, 1] * 100, 0), 100)
    similarities['Doc2Vec'] = min(max(similarity_doc2vec[0, 1] * 100, 0), 100)

    if use_sbert:
        X_combined = np.hstack((X_sbert, X_lsa, X_doc2vec))
    else:
        X_combined = np.hstack((X_lsa, X_doc2vec))

    similarity_matrix_combined = cosine_similarity(X_combined)
    similarity_percentage_combined = min(max(similarity_matrix_combined[0, 1] * 100, 0), 100)

    similarities = {key: max(val, 0) for key, val in similarities.items()}

    if use_sbert:
        logging.info(f"SBERT Similarity Percentage: {similarities['SBERT']:.2f}%")
        print(f"SBERT Similarity Percentage: {similarities['SBERT']:.2f}%")

    logging.info(f"LSA Similarity Percentage: {similarities['LSA']:.2f}%")
    print(f"LSA Similarity Percentage: {similarities['LSA']:.2f}%")

    logging.info(f"Doc2Vec Similarity Percentage: {similarities['Doc2Vec']:.2f}%")
    print(f"Doc2Vec Similarity Percentage: {similarities['Doc2Vec']:.2f}%")

    logging.info(f"Combined Similarity Percentage: {similarity_percentage_combined:.2f}%")
    print(f"Combined Similarity Percentage: {similarity_percentage_combined:.2f}%")

    if use_sbert:
        logging.info("\nCosine Similarity Matrix (SBERT):")
        logging.info(f"{similarity_sbert}")

    logging.info("\nCosine Similarity Matrix (LSA):")
    logging.info(f"{similarity_lsa}")

    logging.info("\nCosine Similarity Matrix (Doc2Vec):")
    logging.info(f"{similarity_doc2vec}")

    logging.info("\nCosine Similarity Matrix (Combined):")
    logging.info(f"{similarity_matrix_combined}")

    jaccard_sim = jaccard_similarity(doc1, doc2)
    logging.info(f"\nJaccard Similarity: {jaccard_sim:.2f}")
    print(f"Jaccard Similarity: {jaccard_sim:.2f}")

    if use_sbert:
        logging.info("\nSVD Components:")
        logging.info(f"U (Document space): {U.shape}")
        logging.info(f"S (Singular values): {S.shape}")
        logging.info(f"Vt (Term space): {Vt.shape}")

    results_dict = {
        'LSA': similarities['LSA'],
        'Doc2Vec': similarities['Doc2Vec'],
        'Combined': similarity_percentage_combined,
        'Jaccard': jaccard_sim * 100
    }

    if use_sbert:
        results_dict['SBERT'] = similarities['SBERT']

    return results_dict

# Validate embeddings for compatibility
def validate_embeddings(X_sbert, X_lsa, X_doc2vec):
    """
    Validate that embeddings from different models have compatible shapes for concatenation.
    """
    if X_sbert is not None:
        if X_sbert.shape[0] != X_lsa.shape[0] or X_sbert.shape[0] != X_doc2vec.shape[0]:
            raise ValueError("Embeddings shapes are not compatible for concatenation.")
    else:
        if X_lsa.shape[0] != X_doc2vec.shape[0]:
            raise ValueError("Embeddings shapes are not compatible for concatenation.")

def compare_document_with_folder(target_file, folder_path, use_sbert=True):
    """
    Compare a target document with each document in a folder and find the one with the highest similarity.
    """
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"The target file at {target_file} does not exist.")
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder at {folder_path} does not exist.")

    max_similarity = -1
    most_similar_file = None
    similarities_dict = {}

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                similarities = compare_documents(target_file, file_path, use_sbert=use_sbert)
                combined_similarity = similarities['Combined']
                similarities_dict[file_path] = similarities
                logging.info(f"Compared {target_file} with {file_path} - Combined Similarity: {combined_similarity:.2f}%")
                print(f"Compared {target_file} with {file_path} - Combined Similarity: {combined_similarity:.2f}%")
                if combined_similarity > max_similarity:
                    max_similarity = combined_similarity
                    most_similar_file = file_path
            except Exception as e:
                logging.error(f"Error comparing {target_file} with {file_path}: {e}")

    if most_similar_file:
        logging.info(f"Most similar document: {most_similar_file} with similarity {max_similarity:.2f}%")
        print(f"Most similar document: {most_similar_file} with similarity {max_similarity:.2f}%")
        return most_similar_file, os.path.basename(most_similar_file), similarities_dict[most_similar_file]
    else:
        logging.info("No similar documents found.")
        print("No similar documents found.")
        return None, None, None


# Modify the main function to include a folder comparison
def main():
    """
    Main function to execute the comparison process.
    """
    target_file = 'test.txt'
    folder_path = 'hasil-skripsi'

    use_sbert = False  # Set to True for higher accuracy

    most_similar_file, most_similar_file_name, similarities = compare_document_with_folder(target_file, folder_path, use_sbert=use_sbert)

    if most_similar_file and similarities:
        print(f"Most similar file: {most_similar_file_name}")
        if use_sbert and 'SBERT' in similarities:
            logging.info(f"\nSBERT Similarity Percentage: {similarities['SBERT']:.2f}%")
            print(f"SBERT Similarity Percentage: {similarities['SBERT']:.2f}%")
        if 'LSA' in similarities:
            logging.info(f"LSA Similarity Percentage: {similarities['LSA']:.2f}%")
            print(f"LSA Similarity Percentage: {similarities['LSA']:.2f}%")
        if 'Doc2Vec' in similarities:
            logging.info(f"Doc2Vec Similarity Percentage: {similarities['Doc2Vec']:.2f}%")
            print(f"Doc2Vec Similarity Percentage: {similarities['Doc2Vec']:.2f}%")
        if 'Combined' in similarities:
            logging.info(f"Combined Similarity Percentage: {similarities['Combined']:.2f}%")
            print(f"Combined Similarity Percentage: {similarities['Combined']:.2f}%")
        if 'Jaccard' in similarities:
            logging.info(f"Jaccard Similarity: {similarities['Jaccard']:.2f}%")
            print(f"Jaccard Similarity: {similarities['Jaccard']:.2f}%")

if __name__ == "__main__":
    main()

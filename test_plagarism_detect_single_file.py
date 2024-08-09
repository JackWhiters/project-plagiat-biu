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
    Get LSA embeddings using TF-IDF and TruncatedSVD.
    """
    indonesian_stopwords_path = 'indonesian_stopwords.txt'

    if os.path.exists(indonesian_stopwords_path):
        with open(indonesian_stopwords_path, 'r', encoding='utf-8') as f:
            stop_words_indonesian = set([word.strip() for word in f.readlines()])
        logging.info(f"Indonesian stopwords loaded: {len(stop_words_indonesian)} words")
    else:
        raise FileNotFoundError("Indonesian stopwords file not found!")

    if os.path.exists(VECTORIZER_PATH) and os.path.exists(SVD_PATH):
        vectorizer = joblib.load(VECTORIZER_PATH)
        svd = joblib.load(SVD_PATH)
    else:
        vectorizer = TfidfVectorizer(
            stop_words=list(stop_words_indonesian),
            max_df=0.95,
            min_df=2,
            ngram_range=(1, 3),
            sublinear_tf=True,
            max_features=100000
        )
        X_tfidf = vectorizer.fit_transform(texts)

        logging.info(f"TF-IDF shape: {X_tfidf.shape}")

        svd = TruncatedSVD(n_components=LSA_COMPONENTS, algorithm='randomized', random_state=42)
        X_lsa = svd.fit_transform(X_tfidf)

        logging.info(f"LSA shape: {X_lsa.shape}")

        joblib.dump(vectorizer, VECTORIZER_PATH)
        joblib.dump(svd, SVD_PATH)

    X_tfidf = vectorizer.transform(texts)
    logging.info(f"Updated TF-IDF shape: {X_tfidf.shape}")
    X_lsa = svd.transform(X_tfidf)
    logging.info(f"Updated LSA shape: {X_lsa.shape}")

    if X_lsa.shape[0] < 2:
        raise ValueError("LSA output does not have enough rows for the comparison!")

    U, S, Vt = np.linalg.svd(X_lsa, full_matrices=False)
    logging.info(f"SVD Shapes: U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")

    if X_lsa.shape[1] < 2:
        raise ValueError("LSA matrix does not have enough components!")

    return X_lsa, svd, U, S, Vt

# Doc2Vec Model
def get_doc2vec_embeddings(texts):
    """
    Get Doc2Vec embeddings for a list of texts.
    """
    if os.path.exists(DOC2VEC_MODEL_PATH):
        model = Doc2Vec.load(DOC2VEC_MODEL_PATH)
        logging.info(f"Loaded existing Doc2Vec model from {DOC2VEC_MODEL_PATH}")
    else:
        tagged_docs = [TaggedDocument(words=preprocess_text(text).split(), tags=[str(i)]) for i, text in enumerate(texts)]
        logging.info(f"Preparing to train a new Doc2Vec model with {len(texts)} documents")

        model = Doc2Vec(vector_size=VECTOR_SIZE, min_count=MIN_COUNT, epochs=EPOCHS, window=WINDOW, dm=DM, workers=4, alpha=ALPHA, min_alpha=MIN_ALPHA)
        model.build_vocab(tagged_docs)
        logging.info(f"Building vocabulary with {len(model.wv)} words")

        for epoch in range(EPOCHS):
            model.train(tagged_docs, total_examples=model.corpus_count, epochs=1)
            model.alpha -= (ALPHA - MIN_ALPHA) / EPOCHS
            model.min_alpha = model.alpha
        logging.info(f"Training complete with {EPOCHS} epochs")

        model.save(DOC2VEC_MODEL_PATH)
        logging.info(f"Saved the Doc2Vec model to {DOC2VEC_MODEL_PATH}")

    embeddings = [model.infer_vector(preprocess_text(text).split(), epochs=EPOCHS) for text in texts]
    embeddings = np.array(embeddings)

    if embeddings.shape[0] < 2:
        raise ValueError("Doc2Vec embeddings do not have enough rows for the comparison!")

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

# Main function to run comparisons
def main():
    """
    Main function to execute the comparison process.
    """
    file1 = 'data1.txt'
    file2 = 'data2.txt'

    use_sbert = False  # Set to True for higher accuracy

    results = {'SBERT': [], 'LSA': [], 'Doc2Vec': [], 'Combined': [], 'Jaccard': []}
    num_runs = 0
    max_runs = 10
    previous_combined_similarity = -1  # Initialize with a value that is different from any valid similarity value

    while num_runs < max_runs:
        num_runs += 1
        logging.info(f"\nRun {num_runs}/{max_runs}")

        sim_results = compare_documents(file1, file2, use_sbert=use_sbert)

        for key in sim_results:
            results[key].append(sim_results[key])
        
        # Check if the combined similarity has changed
        current_combined_similarity = sim_results['Combined']
        if abs(current_combined_similarity - previous_combined_similarity) < 1e-2:  # Check if the difference is below a small threshold
            logging.info(f"Combined similarity has stabilized: {current_combined_similarity:.2f}%")
            break
        
        previous_combined_similarity = current_combined_similarity
    
    # Remove None values before calculating the mean
    avg_results = {}
    for key in results:
        filtered_results = [res for res in results[key] if res is not None]
        if filtered_results:
            avg_results[key] = np.mean(filtered_results)
        else:
            avg_results[key] = None  # Atur rata-rata ke None jika tidak ada data

    if use_sbert and avg_results.get('SBERT') is not None:
        logging.info(f"\nAverage SBERT Similarity Percentage: {avg_results['SBERT']:.2f}%")
        print(f"Average SBERT Similarity Percentage: {avg_results['SBERT']:.2f}%")
    if avg_results.get('LSA') is not None:
        logging.info(f"Average LSA Similarity Percentage: {avg_results['LSA']:.2f}%")
        print(f"Average LSA Similarity Percentage: {avg_results['LSA']:.2f}%")
    if avg_results.get('Doc2Vec') is not None:
        logging.info(f"Average Doc2Vec Similarity Percentage: {avg_results['Doc2Vec']:.2f}%")
        print(f"Average Doc2Vec Similarity Percentage: {avg_results['Doc2Vec']:.2f}%")
    if avg_results.get('Combined') is not None:
        logging.info(f"Average Combined Similarity Percentage: {avg_results['Combined']:.2f}%")
        print(f"Average Combined Similarity Percentage: {avg_results['Combined']:.2f}%")
    if avg_results.get('Jaccard') is not None:
        logging.info(f"Average Jaccard Similarity: {avg_results['Jaccard']:.2f}%")
        print(f"Average Jaccard Similarity: {avg_results['Jaccard']:.2f}%")

if __name__ == "__main__":
    main()

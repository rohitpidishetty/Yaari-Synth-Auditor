📰 Fake-News Classification: Preprocessing & Embedding Pipeline

A complete NLP preprocessing and embedding generation framework designed for fake-news detection on social media. This project handles noisy text, cleans and normalizes it, and generates semantic embeddings using a transformer-based SentenceTransformer model. These embeddings can be directly used for downstream machine-learning experiments.

🚀 Features

Custom text-cleaning pipeline

Large curated stopword list

Lemmatization using WordNet

Transformer-based sentence embeddings (all-mpnet-base-v2)

Saves embeddings and class labels as .pkl files

Fully reproducible for ML workflows

Clean separation between preprocessing and model training

📁 Dataset

The input dataset (_74429.csv) includes:

Column	Description
news	Text from social-media posts
label	Fake (0) or Real (1)
🧹 Preprocessing Pipeline

The preprocessing workflow includes:

Removing special characters

Lowercasing text

Tokenization

Stopword removal (custom-built list)

Lemmatization

Reconstructing cleaned sentences

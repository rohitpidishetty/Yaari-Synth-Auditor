📰 Fake-News Classification: Preprocessing & Embedding Pipeline

A complete NLP preprocessing and embedding generation framework for fake-news detection on social media. This project cleans noisy text data, removes stopwords, performs lemmatization, and generates high-quality semantic embeddings using SentenceTransformer (all-mpnet-base-v2).
The output embeddings are saved for downstream ML models such as SVM, Logistic Regression, and deep neural networks.

🚀 Features

✔ Custom text-cleaning pipeline
✔ Large curated stopword list
✔ Lemmatization using WordNet
✔ Transformer-based embeddings (all-mpnet-base-v2)
✔ Saves embeddings + class labels as .pkl
✔ Fully reproducible workflow
✔ Ready for classification experiments

📁 Dataset

The raw dataset (_74429.csv) contains:

Column	Description
news	Social media post text
label	Fake (0) or Real (1)
🧹 Preprocessing Pipeline

Your preprocessing pipeline performs:

Special Character Removal

Lowercasing

Tokenization

Stopword Removal (custom large stopword list)

Lemmatization

Sentence Reconstruction

Mathematical representation:

𝑇
𝑖
=
Lemmatize
(
RemoveStopwords
(
Lowercase
(
RemoveSpecialChars
(
𝑠
𝑖
)
)
)
)
T
i
	​

=Lemmatize(RemoveStopwords(Lowercase(RemoveSpecialChars(s
i
	​

))))

A cleaned dataset is saved as:

_74429_V01.csv

🔡 Sentence Embeddings

The cleaned text is encoded using:

Model
SentenceTransformer("all-mpnet-base-v2")

Embedding Equation
𝐸
𝑖
=
𝑓
𝜃
(
𝑇
𝑖
)
E
i
	​

=f
θ
	​

(T
i
	​

)

Where:

𝑇
𝑖
T
i
	​

 = cleaned text

𝐸
𝑖
E
i
	​

 = embedding vector

𝑓
𝜃
f
θ
	​

 = transformer model

Output Files

embeddings.pkl — all embedding vectors

embedding_classes.pkl — matching labels

📦 Installation
pip install pandas numpy nltk sentence-transformers tqdm


Download NLTK resources (first run only):

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

▶️ Usage
1. Preprocess the dataset
from preprocess import cleanse
import pandas as pd

data = pd.read_csv("_74429.csv")

cleaned_df = pd.DataFrame({
    "news": data['news'].apply(cleanse),
    "class": data['label']
})

cleaned_df.to_csv("_74429_V01.csv", index=False)

2. Generate embeddings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle

model = SentenceTransformer("all-mpnet-base-v2")

embeddings = []
for text in tqdm(cleaned_df['news']):
    embeddings.append(model.encode(text))

with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

with open("embedding_classes.pkl", "wb") as f:
    pickle.dump(list(cleaned_df['class']), f)

📊 Next Steps / Future Work

Compare different embedding models (BGE, GTE, Gemini)

Fine-tune transformers on fake news datasets

Apply dimensionality reduction (PCA, UMAP)

Build downstream classifiers (SVM, CNN-BiLSTM, Transformers)

📜 Citation

If you use this pipeline, consider citing:

Viswakarma Pidishetti, Rohit.  
"A Preprocessing and Embedding Framework for Social Media Fake-News Classification."

⭐ Contribute

Contributions are welcome!
Feel free to open an issue or submit a pull request.

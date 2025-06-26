
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Percorso della directory dello script
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- 1. Carica il dataset ---
print("File presenti nella cartella:", os.listdir(script_dir))
df = pd.read_csv(os.path.join(script_dir, 'dataset_pronto_per_analisi.csv'))
df_s = df.copy()
print("\nNumero articoli:", len(df))
print("Colonne disponibili:", list(df.columns))

# --- 2. Articoli per rivista ---
print("\nArticoli per rivista:")
print(df['Source title'].value_counts())

# --- 3. Keyword analysis ---
all_keywords = df['Index Keywords'].dropna().str.split(';')
keywords_flat = [kw.strip() for sublist in all_keywords for kw in sublist if isinstance(sublist, list)]
keyword_series = pd.Series(keywords_flat)
top_keywords = keyword_series.value_counts().head(20)
print("\nTop 20 keyword:")
print(list(top_keywords.items()))

# --- 4. Network file (keywords pair) ---
from itertools import combinations
from collections import Counter

keyword_pairs = []

for kw_list in all_keywords:
    cleaned = [kw.strip() for kw in kw_list if kw.strip()]
    pairs = list(combinations(sorted(set(cleaned)), 2))
    keyword_pairs.extend(pairs)

pair_counts = Counter(keyword_pairs)
df_edges = pd.DataFrame(pair_counts.items(), columns=['pair', 'weight'])
df_edges[['source', 'target']] = pd.DataFrame(df_edges['pair'].tolist(), index=df_edges.index)
df_edges = df_edges[['source', 'target', 'weight']]

# Salva file CSV nella directory dello script
df_edges.to_csv(os.path.join(script_dir, 'keys_network.csv'), index=False)
print("✅ File 'keys_network.csv' salvato in:", script_dir)

# --- 5. Preprocessing abstract ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isnull(text): return ""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

df_s['cleaned_abstract'] = df_s['Abstract'].apply(clean_text)

# Frequenze parole
from collections import Counter
all_words = ' '.join(df_s['cleaned_abstract']).split()
word_freq = Counter(all_words)
df_keywords = pd.DataFrame(word_freq.items(), columns=['word', 'count']).sort_values(by='count', ascending=False)

# Salva file CSV nella directory dello script
df_keywords.to_csv(os.path.join(script_dir, 'word_net.csv'), index=False)
print("✅ File 'word_net.csv' salvato in:", script_dir)

# --- 6. Topic Modeling ---
print("\n--- TOPIC MODELING ---")
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(df_s['cleaned_abstract'])

lda = LatentDirichletAllocation(n_components=5, random_state=0)
lda.fit(dtm)

for i, topic in enumerate(lda.components_):
    top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]]
    weights = topic[topic.argsort()[-10:]]
    topic_terms = " + ".join([f"{w:.3f}*\"{t}\"" for t, w in zip(top_words[::-1], weights[::-1])])
    print(f"Topic {i}: {topic_terms}")

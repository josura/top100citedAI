
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from collections import Counter
from itertools import combinations

# Assicura che i pacchetti necessari siano disponibili
nltk.data.path.append("C:/Users/vivia/AppData/Roaming/nltk_data")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

# --- 1. Carica il dataset ---
import os
print("File presenti nella cartella:", os.listdir())
df = pd.read_csv(r'C:\Users\vivia\OneDrive - Università degli Studi di Catania\Desktop\doc dottorato\TESI\cluster miei\dataset_pronto_per_analisi.csv')
df_s = df.copy()
print("\nNumero articoli:", len(df))
print("Colonne disponibili:", df.columns.tolist())

# --- 2. Statistiche base ---
print("\nArticoli per rivista:")
print(df_s['Source title'].value_counts())

# --- 3. Articoli per anno ---
if 'Year' in df_s.columns:
    plt.figure(figsize=(10, 6))
    df_s['Year'].value_counts().sort_index().plot(kind='bar')
    plt.title('Numero di articoli per anno')
    plt.xlabel('Anno')
    plt.ylabel('Numero di articoli')
    plt.tight_layout()
    plt.show()

# --- 4. Keyword Analysis ---
def estrai_keywords(row):
    if pd.isnull(row):
        return []
    return [kw.strip() for kw in str(row).split(';')]

df_s['Index Keywords'] = df_s['Index Keywords'].apply(estrai_keywords)
all_keywords = [kw for kws in df_s['Index Keywords'] for kw in kws]
counter = Counter(all_keywords)
print("\nTop 20 keyword:")
print(counter.most_common(20))

# --- 5. Esportazione rete keyword ---
edges = []
for keywords in df_s['Index Keywords']:
    for a, b in combinations(sorted(set(keywords)), 2):
        edges.append((a, b))

edges_df = pd.DataFrame(edges, columns=['Source', 'Target'])
edges_df.to_csv('keys_network.csv', index=False)
print("✅ File 'keys_network.csv' salvato nella cartella corrente.")

# --- 6. Pulizia testo ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text, preserve_line=True)
    stop_words = set(stopwords.words('english') + ['large','model','language','models','chat','gpt','llm','chatgpt','chatbot'])
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

df_s['cleaned_abstract'] = df_s['Abstract'].apply(clean_text)

# --- 7. Topic Modeling ---
dictionary = corpora.Dictionary(df_s['cleaned_abstract'])
corpus = [dictionary.doc2bow(text) for text in df_s['cleaned_abstract']]
lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)

print("\n--- TOPIC MODELING ---")
for idx, topic in lda.print_topics(num_words=10):
    print(f"Topic {idx}: {topic}")

# --- 8. Esporta co-occorrenze parole in abstract ---
cooc = []
for tokens in df_s['cleaned_abstract']:
    for a, b in combinations(sorted(set(tokens)), 2):
        cooc.append((a, b))

word_net_df = pd.DataFrame(cooc, columns=['Source', 'Target'])
word_net_df.to_csv('word_net.csv', index=False)
print("✅ File 'word_net.csv' salvato nella cartella corrente.")

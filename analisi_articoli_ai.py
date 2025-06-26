
# Analisi articoli con AI - versione semplificata per VSCode

import pandas as pd
import numpy as np
import re
from collections import Counter
from itertools import combinations
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.data.path.append("C:/Users/vivia/AppData/Roaming/nltk_data")
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import matplotlib.pyplot as plt

# --- 1. Carica il dataset ---
import os
print("File presenti nella cartella:", os.listdir())
df = pd.read_csv(r'C:\Users\vivia\OneDrive - Università degli Studi di Catania\Desktop\doc dottorato\TESI\cluster miei\dataset_pronto_per_analisi.csv')
df_s = df.copy()

# --- 2. Analisi base ---
print("Numero articoli:", len(df_s))
print("Colonne disponibili:", df_s.columns.tolist())

# --- 3. Conteggio per rivista ---
print("\nArticoli per rivista:")
print(df_s['Source title'].value_counts())

# --- 4. Grafico articoli per anno ---
df_s['Year'].value_counts().sort_index().plot(kind='bar')
plt.title("Articoli per anno")
plt.xlabel("Anno")
plt.ylabel("Numero di articoli")
plt.tight_layout()
plt.show()

# --- 5. Analisi keyword ---
keylist = df_s['Index Keywords'].dropna().tolist()
keys = []
for row in keylist:
    keys_t = [key.strip() for key in row.split(';')]
    keys.extend(keys_t)

c = Counter(keys)
print("\nTop 20 keyword:")
print(c.most_common(20))

# --- 6. Rete keyword ---
edges_list = []
for row in keylist:
    edges_list.extend(combinations([key.strip() for key in row.split(';')], 2))

edges_df = pd.DataFrame(edges_list, columns=['Source', 'Target'])
edges_df.to_csv('keys_network.csv', index=False)

# --- 7. Topic Modeling ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text, preserve_line=True)
    stop_words = set(stopwords.words('english') + ['large','model','language','models','chat','gpt','llm','chatgpt','chatbot'])
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

df_s['cleaned_abstract'] = df_s['Abstract'].apply(clean_text)

texts = df_s['cleaned_abstract'].tolist()
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda_model = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary, passes=10, random_state=42)

print("\n--- TOPIC MODELING ---")
topics = lda_model.print_topics(num_words=10)
for idx, topic in topics:
    print(f"Topic {idx}: {topic}")

# --- 8. (Opzionale) Rete parole ---
edges_words = []
for row in texts:
    edges_words.extend(combinations(row, 2))

word_net_df = pd.DataFrame(edges_words, columns=['Source', 'Target'])
word_net_df.to_csv('word_net.csv', index=False)

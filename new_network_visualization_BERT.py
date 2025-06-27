import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network

# Dati
df = pd.read_csv('keys_network.csv')
df['weight'] = df.get('weight', 1.0)
nodes = pd.Series(pd.concat([df.Source, df.Target]).unique(), name='node')

# Embedding
model = SentenceTransformer('all-MiniLM-L6-v2')
emb_nodes = model.encode(nodes.tolist())
macro_names = ['Mathematics', 'Biology', 'Computer Science', 'Physics', 'Chemistry']
emb_macros = model.encode(macro_names)

# K-Means micro
num_micro = 10
kmeans = KMeans(n_clusters=num_micro, random_state=42).fit(emb_nodes)
micro_labels = kmeans.labels_
micro_centroids = kmeans.cluster_centers_

# Mappa micro -> macro via similarità coseno
sim = cosine_similarity(micro_centroids, emb_macros)
micro_to_macro = {i: macro_names[sim[i].argmax()] for i in range(num_micro)}
node_to_macro = {nodes[i]: micro_to_macro[micro_labels[i]] for i in range(len(nodes))}

# Costruzione grafo
G = nx.Graph()
for _, r in df.iterrows():
    for n in [r.Source, r.Target]:
        G.add_node(n, macro=node_to_macro.get(n, 'Other'))
    G.add_edge(r.Source, r.Target, weight=r.weight)
centrality = nx.degree_centrality(G)
macros = sorted(set(node_to_macro.values()))

# Palette colori
colors = plt.get_cmap('tab10', len(macros))
col_map = {m: mcolors.to_hex(colors(i)) for i, m in enumerate(macros)}

# Visualizzazione statica
pos = nx.spring_layout(G, seed=42, k=0.5)
plt.figure(figsize=(12,10), dpi=150)
nx.draw_networkx_nodes(G, pos,
                       node_size=[3000*centrality[n] for n in G.nodes()],
                       node_color=[col_map[G.nodes[n]['macro']] for n in G.nodes()],
                       alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=8)
patches = [mpatches.Patch(color=col_map[m], label=m) for m in macros]
plt.legend(handles=patches, title='Macro-categorie', bbox_to_anchor=(1.05,1), loc='upper left')
plt.axis('off')
plt.tight_layout()
plt.savefig('network_macro_sentenceBERT.png', bbox_inches='tight')
plt.show()

# Visualizzazione interattiva
net = Network(height='700px', width='100%', bgcolor='white')
net.toggle_physics(False)
for n in G.nodes():
    net.add_node(n, label=n,
                 color=col_map[G.nodes[n]['macro']],
                 size=10 + 50*centrality[n],
                 group=G.nodes[n]['macro'])
for u, v, d in G.edges(data=True):
    net.add_edge(u, v, value=d['weight'])
net.write_html('network_macro_sentenceBERT.html')

print("✅ Generati PNG e HTML.")

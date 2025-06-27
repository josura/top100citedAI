import pandas as pd, networkx as nx, matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from pyvis.network import Network

# Caricamento dati
df = pd.read_csv('keys_network.csv')
df['weight'] = df.get('weight', 1.0)

# Estrai nodi unici
nodes = pd.Series(pd.concat([df.Source, df.Target]).unique(), name='node')

# TF-IDF su nomi nodi + nomi macro
macro_names = ['Mathematics', 'Biology', 'Computer Science', 'Physics', 'Chemistry', 'Economics',
               'Engineering', 'Geography', 'Sociology', 'History']
vectorizer = TfidfVectorizer(stop_words='english')
X_nodes = vectorizer.fit_transform(nodes)
X_macros = vectorizer.transform(macro_names)

# KMeans micro-cluster
num_micro = 10  # micro cluster
kmeans = KMeans(n_clusters=num_micro, random_state=42, init='k-means++').fit(X_nodes)
micro_labels = kmeans.labels_

# Centroidi micro
micro_centroids = kmeans.cluster_centers_

# Associa micro → macro
dist = cosine_distances(micro_centroids, X_macros)
micro_to_macro = dist.argmin(axis=1)
macro_map = {i: macro_names[j] for i, j in enumerate(micro_to_macro)}

# Mappa nodo → macro
node_macro = {node: macro_map[micro_labels[i]] for i, node in enumerate(nodes)}

# Costruzione grafo
G = nx.Graph()
for _, r in df.iterrows():
    for n in [r.Source, r.Target]:
        G.add_node(n, macro=node_macro.get(n, 'Other'))
    G.add_edge(r.Source, r.Target, weight=r.weight)

centrality = nx.degree_centrality(G)
macros = sorted(set(macro_map.values()))

# Palette colori
colors = plt.get_cmap('tab10', len(macros))
col_map = {m: mcolors.to_hex(colors(i)) for i, m in enumerate(macros)}

# 🎯 Static graph
pos = nx.spring_layout(G, seed=42, k=0.4)
plt.figure(figsize=(12, 10), dpi=150)
nx.draw_networkx_nodes(G, pos, node_size=[3000*centrality[n] for n in G],
                       node_color=[col_map[G.nodes[n]['macro']] for n in G], alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=8)
patches = [mpatches.Patch(color=col_map[m], label=m) for m in macros]
plt.legend(handles=patches, title='Macro-categories', bbox_to_anchor=(1.05,1), loc='upper left')
plt.axis('off')
plt.tight_layout()
plt.savefig('network_macro_kmeans.png', bbox_inches='tight')
plt.show()

# 🌐 Interactive graph
net = Network(height='700px', width='100%', bgcolor='white')
net.toggle_physics(False)
for n in G.nodes():
    net.add_node(n, label=n, size=10+50*centrality[n],
                 color=col_map[G.nodes[n]['macro']],
                 group=G.nodes[n]['macro'])
for u, v, d in G.edges(data=True):
    net.add_edge(u, v, value=d['weight'])
net.write_html('network_macro_kmeans_dynamic.html')
print("✅ Creati PNG e HTML con macro-categoria")

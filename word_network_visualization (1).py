
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

# Imposta la directory di lavoro
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Caricamento dati
wordnet_df = pd.read_csv('word_net.csv')

# Se non c'è la colonna 'weight', impostala a 1
if 'weight' not in wordnet_df.columns:
    wordnet_df['weight'] = 1.0

# Filtro per co-occorrenze più forti
threshold = wordnet_df['weight'].quantile(0.75)
filtered_df = wordnet_df[wordnet_df['weight'] >= threshold]

# Costruzione grafo
G = nx.Graph()
for _, row in filtered_df.iterrows():
    G.add_edge(row['Source'], row['Target'], weight=row['weight'])

# Calcolo centralità
centrality = nx.degree_centrality(G)

# --- GRAFICO STATICO ---
plt.figure(figsize=(18, 14))
pos = nx.spring_layout(G, k=0.4, iterations=100, seed=42)
node_sizes = [3000 * centrality[n] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightcoral', alpha=0.8)
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)

# Etichette centrali
central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:30]
labels = {node: node for node, _ in central_nodes}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

plt.title('Rete parole da abstract (etichette centrali)')
plt.axis('off')
plt.tight_layout()
plt.savefig('word_network_static.png', dpi=300)
plt.show()

# --- GRAFICO INTERATTIVO ---
net = Network(height='750px', width='100%', bgcolor='white', font_color='black')
net.barnes_hut()

for node in G.nodes():
    net.add_node(node, label=node, title=f"Centralità: {centrality[node]:.3f}",
                 size=10 + 40 * centrality[node], color='lightcoral')

for source, target, data in G.edges(data=True):
    net.add_edge(source, target, value=data['weight'])

net.write_html('word_network_interactive.html')
print("✅ File 'word_network_static.png' e 'word_network_interactive.html' creati dalla rete di parole.")

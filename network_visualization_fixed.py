
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

# Imposta la directory dello script come directory di lavoro
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Carica i dati
keys_df = pd.read_csv('keys_network.csv')
words_df = pd.read_csv('word_net.csv')

# Se manca 'weight', assegna 1.0 a tutte le connessioni
if 'weight' not in keys_df.columns:
    keys_df['weight'] = 1.0

# Se manca 'weight' anche in word_net.csv
if 'weight' not in words_df.columns:
    words_df['weight'] = 1.0

# Costruisci il grafo
G = nx.Graph()
for _, row in keys_df.iterrows():
    G.add_edge(row['Source'], row['Target'], weight=row['weight'])

# Calcolo centralità
centrality = nx.degree_centrality(G)

# --- Visualizzazione statica ---
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, seed=42)
node_sizes = [5000 * centrality[n] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=6)
plt.title('Keyword Co-occurrence Network (Static)')
plt.tight_layout()
plt.savefig('network_static.png')
plt.show()

# --- Visualizzazione interattiva ---
net = Network(height='750px', width='100%', notebook=False)
net.barnes_hut()

for node in G.nodes():
    net.add_node(node, label=node, size=10 + 40 * centrality[node])

for source, target, data in G.edges(data=True):
    net.add_edge(source, target, value=data['weight'])

net.write_html('network_interactive.html')

print("✅ File 'network_static.png' e 'network_interactive.html' creati correttamente.")

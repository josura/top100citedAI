
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

# Directory dello script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Caricamento dati
keys_df = pd.read_csv('keys_network.csv')
words_df = pd.read_csv('word_net.csv')

# Assicurati che esistano le colonne necessarie
if 'weight' not in keys_df.columns:
    keys_df['weight'] = 1.0
if 'weight' not in words_df.columns:
    words_df['weight'] = 1.0
if 'cluster' not in keys_df.columns:
    keys_df['cluster'] = 'default'

# Crea grafo
G = nx.Graph()

# Crea mapping cluster per colore
cluster_map = keys_df.set_index('Source')['cluster'].to_dict()
all_clusters = list(set(cluster_map.values()))
color_palette = plt.cm.get_cmap('tab20', len(all_clusters))
cluster_colors = {cl: f"#{''.join(format(int(c*255), '02x') for c in color_palette(i)[:3])}" for i, cl in enumerate(all_clusters)}

# Aggiunge nodi
for node in pd.concat([keys_df['Source'], keys_df['Target']]).unique():
    cluster = cluster_map.get(node, 'default')
    G.add_node(node, cluster=cluster)

# Aggiunge archi
for _, row in keys_df.iterrows():
    G.add_edge(row['Source'], row['Target'], weight=row['weight'])

# Centralità
centrality = nx.degree_centrality(G)

# --- STATICO (etichette centrali, colore unico) ---
plt.figure(figsize=(18, 14))
pos = nx.spring_layout(G, k=0.4, iterations=100, seed=42)
node_sizes = [3000 * centrality[n] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)

central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:30]
labels = {node: node for node, _ in central_nodes}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

plt.title('Rete parole chiave - statico (etichette centrali)')
plt.axis('off')
plt.tight_layout()
plt.savefig('network_static_readable.png', dpi=300)
plt.show()

# --- INTERATTIVO CON COLORI PER CLUSTER ---
net = Network(height='750px', width='100%', bgcolor='white', font_color='black')
net.barnes_hut()

for node in G.nodes(data=True):
    c = node[1]['cluster']
    net.add_node(node[0], label=node[0], title=f"Cluster: {c}, Centralità: {centrality[node[0]]:.3f}",
                 size=10 + 40 * centrality[node[0]], color=cluster_colors.get(c, "#999999"))

for source, target, data in G.edges(data=True):
    net.add_edge(source, target, value=data['weight'])

net.write_html('network_interactive_colored.html')
print("✅ File 'network_static_readable.png' e 'network_interactive_colored.html' creati con colori per cluster.")

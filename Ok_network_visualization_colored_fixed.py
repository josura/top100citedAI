
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

# Imposta directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Carica CSV
df = pd.read_csv('keys_network.csv')
if 'weight' not in df.columns:
    df['weight'] = 1.0
if 'cluster' not in df.columns:
    df['cluster'] = 'default'

# Costruisci grafo
G = nx.Graph()
for _, row in df.iterrows():
    G.add_node(row['Source'], cluster=row['cluster'])
    G.add_node(row['Target'], cluster=row['cluster'])
    G.add_edge(row['Source'], row['Target'], weight=row['weight'])

# Centralità e cluster
centrality = nx.degree_centrality(G)
clusters = list(set(nx.get_node_attributes(G, 'cluster').values()))
import matplotlib.cm as cm
import matplotlib.colors as mcolors
color_map = cm.get_cmap('tab20', len(clusters))
cluster_colors = {cl: mcolors.to_hex(color_map(i)) for i, cl in enumerate(clusters)}

# --- STATICO ---
plt.figure(figsize=(20, 16), dpi=400)
pos = nx.spring_layout(G, seed=42, k=0.4)
node_sizes = [3000 * centrality[n] for n in G.nodes()]
node_colors = [cluster_colors[G.nodes[n]['cluster']] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.2)

# Etichette dinamiche per i top 50
top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:50]
font_sizes = {n: 4 + 10 * centrality[n] for n, _ in top_nodes}
for node, _ in top_nodes:
    x, y = pos[node]
    plt.text(x, y, node, fontsize=font_sizes[node], ha='center', va='center')

plt.title('Rete keyword - cluster colorati, etichette proporzionali')
plt.axis('off')
plt.tight_layout()
plt.savefig('network_static_colored.png')
plt.show()

# --- INTERATTIVO ---
net = Network(height='800px', width='100%', bgcolor='white', font_color='black')
net.barnes_hut()

for node in G.nodes():
    cl = G.nodes[node]['cluster']
    net.add_node(node, label=node, title=f"Cluster: {cl}, Centralità: {centrality[node]:.3f}",
                 size=10 + 40 * centrality[node], color=cluster_colors[cl])

for src, tgt, data in G.edges(data=True):
    net.add_edge(src, tgt, value=data['weight'])

net.write_html('network_interactive_colored.html')
print("✅ File 'network_static_colored.png' e 'network_interactive_colored.html' aggiornati.")

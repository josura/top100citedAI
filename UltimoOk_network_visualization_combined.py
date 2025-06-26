import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# --- CONFIGURAZIONE ---
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)

# Caricamento dati
df = pd.read_csv('keys_network.csv')
if 'weight' not in df.columns:
    df['weight'] = 1.0
if 'cluster' not in df.columns:
    df['cluster'] = 'default'

# Etichette testuali (modifica a piacere)
cluster_labels = {
    '0': 'Formazione e Didattica',
    '1': 'Etica e Normative',
    '2': 'Tecnologia e Sviluppo',
    '3': 'Interazione Uomo-Macchina',
    '4': 'Applicazioni Sociali',
    'default': 'Altro'
}

# Mappa cluster per nodo
source_clusters = df[['Source', 'cluster']].rename(columns={'Source': 'node'})
target_clusters = df[['Target', 'cluster']].rename(columns={'Target': 'node'})
all_nodes = pd.concat([source_clusters, target_clusters], ignore_index=True).drop_duplicates('node')
node_cluster_map = dict(zip(all_nodes['node'], all_nodes['cluster']))
clusters = sorted(set(node_cluster_map.values()))

# Colori per cluster
color_map = plt.get_cmap('tab20', len(clusters))
cluster_colors = {cl: mcolors.to_hex(color_map(i)) for i, cl in enumerate(clusters)}

# --- COSTRUISCI GRAFO ---
G = nx.Graph()
for _, row in df.iterrows():
    src, tgt = row['Source'], row['Target']
    for node in [src, tgt]:
        G.add_node(node, cluster=node_cluster_map.get(node, 'default'))
    G.add_edge(src, tgt, weight=row['weight'])

centrality = nx.degree_centrality(G)

# --- VISUALIZZAZIONE STATICA ---
plt.figure(figsize=(20, 16), dpi=400)
pos = nx.spring_layout(G, seed=42, k=0.4)
node_sizes = [3000 * centrality[n] for n in G.nodes()]
node_colors = [cluster_colors[G.nodes[n]['cluster']] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.2)

# Etichette top nodi
top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:50]
font_sizes = {n: 4 + 10 * centrality[n] for n, _ in top_nodes}
for node, _ in top_nodes:
    x, y = pos[node]
    plt.text(x, y, node, fontsize=font_sizes[node], ha='center', va='center')

# Etichette cluster (centroide)
for cl in clusters:
    nodes_in_cluster = [n for n in G.nodes() if G.nodes[n]['cluster'] == cl]
    if nodes_in_cluster:
        x_coords = [pos[n][0] for n in nodes_in_cluster]
        y_coords = [pos[n][1] for n in nodes_in_cluster]
        plt.text(sum(x_coords)/len(x_coords),
                 sum(y_coords)/len(y_coords),
                 cluster_labels.get(str(cl), f"Cluster {cl}"),
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
                 fontsize=10, ha='center', va='center')

plt.title('Rete keyword - statica con cluster e etichette')
plt.axis('off')
plt.tight_layout()
plt.savefig('network_static_with_labels.png')
plt.show()

# --- VISUALIZZAZIONE INTERATTIVA ---
net = Network(height='850px', width='100%', bgcolor='white', font_color='black')
net.barnes_hut()

net.set_options("""{
  "nodes": {
    "font": {
      "size": 20,
      "face": "arial"
    },
    "scaling": {
      "min": 10,
      "max": 40
    }
  },
  "edges": {
    "color": {
      "inherit": true
    },
    "smooth": {
      "type": "dynamic"
    }
  },
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -8000,
      "centralGravity": 0.3,
      "springLength": 100
    },
    "stabilization": {
      "iterations": 250
    }
  }
}""")

for node in G.nodes():
    cl = G.nodes[node]['cluster']
    net.add_node(node,
                 label=node,
                 title=f"Cluster: {cluster_labels.get(str(cl), f'Cluster {cl}')}\nCentralità: {centrality[node]:.3f}",
                 size=10 + 50 * centrality[node],
                 color=cluster_colors.get(cl, "#cccccc"),
                 group=str(cl))

for src, tgt, data in G.edges(data=True):
    net.add_edge(src, tgt, value=data['weight'])

net.write_html('network_interactive_with_labels.html')
print("✅ Generati: network_static_with_labels.png e network_interactive_with_labels.html")

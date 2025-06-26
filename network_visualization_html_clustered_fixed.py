
import os
import pandas as pd
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Imposta directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Carica il dataset
df = pd.read_csv('keys_network.csv')
if 'weight' not in df.columns:
    df['weight'] = 1.0
if 'cluster' not in df.columns:
    df['cluster'] = 'default'

# Costruzione grafo
G = nx.Graph()

# Cluster da nodi
source_clusters = df[['Source', 'cluster']].rename(columns={'Source': 'node'})
target_clusters = df[['Target', 'cluster']].rename(columns={'Target': 'node'})
all_nodes = pd.concat([source_clusters, target_clusters], ignore_index=True).drop_duplicates('node')
node_cluster_map = dict(zip(all_nodes['node'], all_nodes['cluster']))
clusters = sorted(set(node_cluster_map.values()))

# Colori cluster
color_map = plt.get_cmap('tab20', len(clusters))
cluster_colors = {cl: mcolors.to_hex(color_map(i)) for i, cl in enumerate(clusters)}

# Nodi e archi
for _, row in df.iterrows():
    src, tgt = row['Source'], row['Target']
    for node in [src, tgt]:
        G.add_node(node, cluster=node_cluster_map.get(node, 'default'))
    G.add_edge(src, tgt, weight=row['weight'])

# Centralità
centrality = nx.degree_centrality(G)

# --- VISUALIZZAZIONE INTERATTIVA ---
net = Network(height='850px', width='100%', bgcolor='white', font_color='black')
net.barnes_hut()

# Opzioni JSON valide
net.set_options("""{
  "nodes": {
    "font": {
      "size": 20,
      "face": "arial",
      "vadjust": 0
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

# Aggiungi nodi
for node in G.nodes():
    cl = G.nodes[node]['cluster']
    net.add_node(node,
                 label=node,
                 title=f"Cluster: {cl}\nCentralità: {centrality[node]:.3f}",
                 size=10 + 50 * centrality[node],
                 color=cluster_colors.get(cl, "#cccccc"),
                 group=cl)

# Aggiungi archi
for src, tgt, data in G.edges(data=True):
    net.add_edge(src, tgt, value=data['weight'])

net.write_html('network_interactive_clustered_fixed.html')
print("✅ File 'network_interactive_clustered_fixed.html' generato correttamente.")


import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

# Imposta la directory corrente
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Carica i file CSV
keys_df = pd.read_csv('keys_network.csv')
words_df = pd.read_csv('word_net.csv')

# Filtro automatico: soglia sul peso
threshold = keys_df['weight'].quantile(0.75)
filtered_keys = keys_df[keys_df['weight'] >= threshold]

# Crea grafo NetworkX
G = nx.Graph()

# Aggiunge nodi con attributi di cluster
for _, row in words_df.iterrows():
    G.add_node(row['key'], cluster=row['cluster'])

# Aggiunge archi filtrati
for _, row in filtered_keys.iterrows():
    if G.has_node(row['source']) and G.has_node(row['target']):
        G.add_edge(row['source'], row['target'], weight=row['weight'])

# --- RETE STATICA ---
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=0.3, seed=42)
clusters = [G.nodes[n]['cluster'] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=150, node_color=clusters, cmap=plt.cm.tab10)
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.axis('off')
plt.tight_layout()
plt.savefig('network_static.png')
plt.close()

# --- RETE INTERATTIVA ---
net = Network(height='800px', width='100%', notebook=False)
net.barnes_hut()
for node in G.nodes(data=True):
    net.add_node(node[0], label=node[0], title=f"Cluster: {node[1]['cluster']}", group=node[1]['cluster'])
for source, target, data in G.edges(data=True):
    net.add_edge(source, target, value=data['weight'])
net.show('network_interactive.html')

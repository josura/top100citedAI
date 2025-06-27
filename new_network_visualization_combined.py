import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Loading data
df = pd.read_csv('keys_network.csv')
if 'weight' not in df.columns:
    df['weight'] = 1.0

# Extract node names
all_nodes = pd.concat([
    df[['Source']].rename(columns={'Source': 'node'}),
    df[['Target']].rename(columns={'Target': 'node'})
]).drop_duplicates('node').reset_index(drop=True)

# NLP Classification using KMeans and TF-IDF
num_categories = 10  # Initial fine-grained clusters
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(all_nodes['node'])

kmeans = KMeans(n_clusters=num_categories, random_state=42)
all_nodes['fine_cluster'] = kmeans.fit_predict(X)

# Define macro-categories mapping
macro_category_map = {
    0: 'Mathematics',
    1: 'Biology',
    2: 'Computer Science',
    3: 'Physics',
    4: 'Chemistry',
    5: 'Engineering',
    6: 'Medicine',
    7: 'Economics',
    8: 'Humanities',
    9: 'Social Sciences'
}

all_nodes['macro_cluster'] = all_nodes['fine_cluster'].map(macro_category_map)
node_cluster_map = dict(zip(all_nodes['node'], all_nodes['macro_cluster']))
macro_clusters = sorted(set(macro_category_map.values()))

# Colors for macro-clusters
color_map = plt.get_cmap('tab10', len(macro_clusters))
cluster_colors = {cl: mcolors.to_hex(color_map(i)) for i, cl in enumerate(macro_clusters)}

# Construct the graph
G = nx.Graph()
for _, row in df.iterrows():
    G.add_node(row['Source'], cluster=node_cluster_map.get(row['Source']))
    G.add_node(row['Target'], cluster=node_cluster_map.get(row['Target']))
    G.add_edge(row['Source'], row['Target'], weight=row['weight'])

centrality = nx.degree_centrality(G)

# Static visualization with legend
plt.figure(figsize=(20, 16), dpi=300)
pos = nx.spring_layout(G, seed=42, k=0.4)
node_sizes = [3000 * centrality[n] for n in G.nodes()]
node_colors = [cluster_colors[G.nodes[n]['cluster']] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=8)

patches = [mpatches.Patch(color=cluster_colors[cl], label=cl) for cl in macro_clusters]
plt.legend(handles=patches, title="Macro-categories", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.axis('off')
plt.tight_layout()
plt.savefig('network_macro_clusters.png', bbox_inches='tight')
plt.show()

# Interactive stable visualization
# net = Network(height='850px', width='100%', bgcolor='white')
# net.toggle_physics(False)

# Interactive visualization with initial auto-spacing
net = Network(height='850px', width='100%', bgcolor='white')

for node in G.nodes():
    cl = G.nodes[node]['cluster']
    net.add_node(node,
                 label=node,
                 size=10 + 50 * centrality[node],
                 color=cluster_colors.get(cl, "#cccccc"),
                 group=str(cl))

for src, tgt, data in G.edges(data=True):
    net.add_edge(src, tgt, value=data['weight'])

net.write_html('network_macro_clusters_stable.html')
print("✅ Generated: network_macro_clusters.png and network_macro_clusters_stable.html")

# TODO classify the nodes into clusters (with names depending on the context of the node name)
# TODO to do this, we need to NLP from the node names to a number of fields (Mathematical analysis, Optimization, Mathematical modeling and other subjects connected to the mathematical field should be classified into the same cluster with the name "Mathematics")
# TODO Cancer, Biology, Genetics, and other subjects connected to the biological field should be classified into the same cluster with the name "Biology"

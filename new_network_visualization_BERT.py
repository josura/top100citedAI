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
# Le macro-categorie sono definite manualmente, aggiungile se ti servono
macro_names = ['Mathematics', 'Biology', 'Computer Science', 'Physics', 'Chemistry', 'Economics',
               'Engineering', 'Geography', 'Sociology', 'History']
emb_macros = model.encode(macro_names)

# Classificazione diretta
sim = cosine_similarity(emb_nodes, emb_macros)
best_macro = sim.argmax(axis=1)
node_to_macro = {nodes[i]: macro_names[best_macro[i]] for i in range(len(nodes))}

# Salvataggio macro
macro_df = pd.DataFrame(list(node_to_macro.items()), columns=['node', 'macro'])
macro_df.to_csv('macro_mapping.csv', index=False)

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
pos = nx.spring_layout(G, seed=42, k=1.5)
plt.figure(figsize=(20,15), dpi=300)
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
#net.toggle_physics(False)
# net.show_buttons(filter_=['physics'])
net.barnes_hut()

options = """{
  "physics": {
    "barnesHut": {
      "theta": 0.35,
      "gravitationalConstant": -1000,
      "centralGravity": 0.5,
      "springLength": 355,
      "springConstant": 0.035,
      "damping": 0.47
    },
    "minVelocity": 0.75
  }
}"""
net.set_options(options)
# Aggiungere nodi
for n in G.nodes():
    net.add_node(n, label=n,
                 color=col_map[G.nodes[n]['macro']],
                 size=10 + 50*centrality[n],
                 group=G.nodes[n]['macro'])
# Aggiungere archi
for u, v, d in G.edges(data=True):
    net.add_edge(u, v, value=d['weight'])
# Scrivere l'output HTML
net.write_html('network_macro_sentenceBERT.html')

# Generazione leggenda come barra laterale nell'html
html_path = 'network_macro_sentenceBERT.html'
net.write_html(html_path)

# Inserisci manualmente la legenda in HTML
with open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Costruzione blocco legenda CSS+HTML
legend_html = "<div style='position: absolute; top: 20px; right: 20px; background: white; border: 1px solid #ccc; padding: 10px; z-index:9999;'>"
legend_html += "<b>Macro-categories</b><br>"
for macro in macros:
    color = col_map[macro]
    legend_html += f"<div style='margin:4px'><span style='display:inline-block;width:12px;height:12px;background:{color};margin-right:6px'></span>{macro}</div>"
legend_html += "</div>"

# Inserisci prima di </body>
html = html.replace("</body>", legend_html + "\n</body>")

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)


print("✅ Generati PNG e HTML.")

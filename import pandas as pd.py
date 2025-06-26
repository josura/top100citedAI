import pandas as pd

# Carica il file
df = pd.read_csv("keys_network.csv")

# Mostra le prime righe e le colonne
print("Colonne:", df.columns)
print("\nPrime righe:\n", df.head())

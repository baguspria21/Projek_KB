import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

data = {
    'Daerah': ['Palu', 'Donggala', 'Poso', 'Morowali', 'Tolitoli', 'Banggai', 'Sigi', 'Parigi'],
    'Jumlah_Penduduk': [300000, 80000, 100000, 70000, 50000, 150000, 45000, 60000],
    'Penggunaan_Air_PerOrang': [150, 130, 140, 160, 120, 140, 110, 135],
    'Musim_Kemarau': [1, 0, 1, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

X = df[['Jumlah_Penduduk', 'Penggunaan_Air_PerOrang', 'Musim_Kemarau']].values

kmeans = KMeans(n_clusters=2)
df['Cluster'] = kmeans.fit_predict(X)

df['Kebutuhan_Air_Total'] = df['Jumlah_Penduduk'] * df['Penggunaan_Air_PerOrang'] * (1 + df['Musim_Kemarau'] * 0.2)

print(df[['Daerah', 'Cluster', 'Kebutuhan_Air_Total']])

print("\nPusat Cluster: ")
print(kmeans.cluster_centers_)

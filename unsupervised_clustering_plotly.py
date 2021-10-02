#Plotly
import plotly.figure_factory as ff
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd

customer_data = pd.read_csv('/content/shopping-data.csv')

#Features
data = customer_data.iloc[:, 3:5].values
#Labels transformations
label= customer_data.iloc[:,1:2]
label_final = label.values.tolist()
#label = ['Male', 'Male', 'Female', 'Female', 'Female']

#Fit
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)
#Plot
fig = ff.create_dendrogram(data, orientation='left', labels=label_final)
fig.update_layout(width=800, height=800)
fig.show()

label_final

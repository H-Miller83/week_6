# %%
# load libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# %%
# load data
# help(pd.read_csv)
df = pd.read_csv("house_votes_Dem.csv", encoding='latin-1')
# %%
# take a look at the data
# df.head()
df.info()

# %%
# separate out the numeric features
numeric = df[['aye','nay','other']]

# %%
# documentation for kmeans in sklearn
help(KMeans)


# %% build a kmeans model
model = KMeans(n_clusters=3, random_state=42, verbose=1)   # random_state ensures centroids randomly start in same spot
model.fit(numeric)


# %% look at the information in the model
print(model.cluster_centers_)
print(model.labels_)

# %%
# add the cluster labels to the original data frame
df['cluster'] = model.labels_
df.head()

# %%
  
# %% simple plot of the clusters
help(plt.scatter)
 
# %%
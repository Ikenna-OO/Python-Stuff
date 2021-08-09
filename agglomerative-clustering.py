# import the packages
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster 

#some initializations
ac = AgglomerativeClustering(n_clusters = 8, affinity = "euclidean", linkage="average")
X, y = make_blobs(n_samples = 1000, centers=8, n_features=2, random_state=800)
distances = linkage(X, method="centroid", metric="euclidean")
sklearn_clusters = ac.fit_predict(X)
scipy_clusters = fcluster(distances, 3, criterion="distance")

#show me some stuff
plt.figure(figsize=(6,4))
plt.title("Clusters from Sci-Kit Learn Approach")
plt.scatter(X[:, 0], X[:, 1], c = sklearn_clusters, s=50, cmap='tab20b')
plt.show()
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import numpy as np


dataset = fetch_20newsgroups(subset='all',shuffle=True)

labels = dataset.target
true_k = np.unique(labels).shape[0]

vectorizer = TfidfVectorizer(max_df=0.5, stop_words='english')
X = vectorizer.fit_transform(dataset.data)

km_unique_labels = MiniBatchKMeans(n_clusters=true_k,
                                   init='k-means++', n_init=1,
                                   init_size=1000, batch_size=1000,
                                   verbose=False,
                                   reallocation_type='unique_labels',
                                   reassignment_ratio=0.30)

km_nonunique_labels = MiniBatchKMeans(n_clusters=true_k,
                                      init='k-means++', n_init=1,
                                      init_size=1000, batch_size=1000,
                                      verbose=False,
                                      reallocation_type='nonunique_labels',
                                      reassignment_ratio=0.30)

km_unique_uniform_labels = MiniBatchKMeans(n_clusters=true_k,
                                           init='k-means++', n_init=1,
                                           init_size=1000, batch_size=1000,
                                           verbose=False,
                                           reallocation_type='unique_uniform_labels',
                                           reassignment_ratio=0.30)

km_nonunique_uniform_labels = MiniBatchKMeans(n_clusters=true_k,
                                              init='k-means++', n_init=1,
                                              init_size=1000, batch_size=1000,
                                              verbose=False,
                                              reallocation_type='nonunique_uniform_labels',
                                              reassignment_ratio=0.30)

iters = 10
km_unique_labels_inertia = np.array([
    km_unique_labels.fit(X).inertia_
    for x in range(iters)])

km_nonunique_labels_inertia = np.array([
    km_nonunique_labels.fit(X).inertia_
    for x in range(iters)])

km_unique_uniform_labels_inertia = np.array([
    km_unique_uniform_labels.fit(X).inertia_
    for x in range(iters)])

km_nonunique_uniform_labels_inertia = np.array([
    km_nonunique_uniform_labels.fit(X).inertia_
    for x in range(iters)])

print(km_unique_labels_inertia)
print(km_nonunique_labels_inertia)
print(km_unique_uniform_labels_inertia)
print(km_nonunique_uniform_labels_inertia)

print('km_unique_labels_inertia average: ',
      km_unique_labels_inertia.mean() ,', std: ',
      km_unique_labels_inertia.std())

print('km_nonunique_labels_inertia average: ',
      km_nonunique_labels_inertia.mean() ,', std: ',
      km_nonunique_labels_inertia.std())

print('km_unique_uniform_labels_inertia average: ',
      km_unique_uniform_labels_inertia.mean() ,', std: ',
      km_unique_uniform_labels_inertia.std())

print('km_nonunique_uniform_labels_inertia average: ',
      km_nonunique_uniform_labels_inertia.mean() ,', std: ',
      km_nonunique_uniform_labels_inertia.std())

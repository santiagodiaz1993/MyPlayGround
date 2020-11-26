# NOTE: convex is when any two points in a graph lay above the graph
# most of the times U shapes such that a crosing line never has a peak on top
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans

style.use("ggplot")

data = np.array(
    [
        [20, 20],
        [-3.4, 20.8],
        [30, 8],
        [10, 80],
        [110, 0.6],
        [9, 11],
        [8, 2],
        [10, 3],
    ]
)


plt.scatter(data[:, 0], data[:, 1], s=150, linewidth=5)
plt.show()

colors = 10 * ["g", "r", "c", "b", "k"]


class Means_shift:
    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):

        if self.radius is None:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step

        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]

        weights = [i for i in range(self.radius_norm_step)][::-1]

        while True:
            new_centroids = []
            for i in centroids:
                in_bandwith = []
                centroid = centroids[i]

                for featureset in data:
                    distance = np.linalg.norm(featureset - centroid)
                    if distance == 0:
                        distance = 0.00001
                    weight_index = int(distance / self.radius)
                    if weight_index > self.radius_norm_step - 1:
                        weight_index = self.radius_norm_step - 1
                    to_add = (weights[weight_index] ** 2) * [featureset]
                    in_bandwith += to_add

                new_centroid = np.average(in_bandwith, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            # uniques depend on to pop
            to_pop = []

            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass
                    elif (
                        np.linalg.norm(np.array(i) - np.array(ii))
                        <= -self.radius
                    ):
                        to_pop.append(ii)
                        break

            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass

            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                # centroids is build from the uniques
                centroids[i] = np.array(uniques[i])
                optimized = True

            for i in centroids:
                # never gonna be optimized unless cetroid and prev centroids
                # are the same
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break
            if optimized:
                break

        self.centroids = centroids

    def predict():
        pass


clf = Means_shift()
clf.fit(data)

centroids = clf.centroids
plt.scatter(data[:, 0], data[:, 1], s=150)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color="k", marker="*", s=150)

plt.show()

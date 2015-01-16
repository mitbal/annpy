import math
import random
import matplotlib.pyplot as plt


class ANN:

    def __init__(self, train_set, k, num_buckets):
        self.train_set = train_set
        self.k = k
        self.num_data = len(train_set)
        self.dim = len(train_set[0])-1

        self.num_buckets = num_buckets
        self.buckets = [{} for x in xrange(num_buckets)]
        self.projections = [[] for x in xrange(num_buckets)]

        # Preprocessing of training input
        for l in xrange(num_buckets):
            # Generate random projection
            self.num_proj = 5
            proj = []
            for i in xrange(self.num_proj):
                cur_proj = []
                for j in xrange(self.dim):
                    cur_proj += [random.random()]
                proj += [cur_proj]
            self.projections[l] = proj

            for x in train_set:
                key = self.lsh(x, self.projections[l])
                if key not in self.buckets[l].keys():
                    self.buckets[l][key] = [x]
                else:
                    self.buckets[l][key] += [x]

    # Calculate dot product between two vectors
    # Used as component for hash function for cosine distance
    def dot_product(self, a, b):
        temp_sum = 0
        for i in xrange(self.dim):
            temp_sum += a[i]*b[i]
        return temp_sum

    # Compute lsh for cosine distance measure
    def lsh(self, x, proj):
        key = ''
        for p in proj:
            dot = self.dot_product(x, p)
            if dot > 0:
                key += '1'
            else:
                key += '0'
        return key

    # Given an instance, get the label from near nearest neighbor in the training set
    def predict(self, a):
        neighbors = self.get_approximate_neighbors(a)
        vote = [0]*2
        for n in neighbors:
            vote[int(n[-1])] += 1
        return vote.index(max(vote))

    # Calculate euclidean distance between two points
    def calc_dist(self, a, b):
        temp_sum = 0
        for i in xrange(self.dim):
            temp_sum += math.pow(a[i] - b[i], 2)
        return math.sqrt(temp_sum)

    # Get k approximately nearest neighbor
    def get_approximate_neighbors(self, a):
        candidates = []
        for l in xrange(self.num_buckets):
            key = self.lsh(a, self.projections[l])
            candidates += self.buckets[l][key]
        neighbors = []
        distances = []
        for c in candidates:
            distances += [(c, self.calc_dist(a, c))]
        distances.sort(key= lambda tup: tup[1])
        for i in xrange(self.k):
            neighbors += [distances[i][0]]
        return neighbors

    # Plot the first two dimension of training dataset
    def plot(self, dataset):
        plt.plot([x[0] for x in dataset if x[2] == 0], [x[1] for x in dataset if x[2] == 0], 'ro')
        plt.plot([x[0] for x in dataset if x[2] == 1], [x[1] for x in dataset if x[2] == 1], 'b^')

    # Plot the data inside each bucket
    def plot_buckets(self):
        n = len(self.buckets.keys())
        index = 1
        for k in self.buckets.keys():
            data = self.buckets[k]
            plt.subplot(n/2, 2, index)
            index += 1
            self.plot(data)
        plt.show()

import math


class NN:
    def __init__(self, train_set, k):
        self.train_set = train_set
        self.k = k

        self.num_train_data = len(train_set)
        self.dim = len(train_set[0]) - 1

    def predict(self, data):
        neighbors = self.get_exact_neighbors(data)
        vote = [0]*2
        for n in neighbors:
            vote[int(n[-1])] += 1
        return vote.index(max(vote))

    def calc_dist(self, a, b):
        temp_sum = 0
        for i in xrange(self.dim):
            temp_sum += math.pow(a[i] - b[i], 2)
        return math.sqrt(temp_sum)

    def get_exact_neighbors(self, a):
        distances = []
        for i in xrange(self.num_train_data):
            b = self.train_set[i]
            distances += [(i, self.calc_dist(a, b))]

        distances.sort(key=lambda tup: tup[1])
        neighbors = []
        for i in xrange(self.k):
            neighbors += [self.train_set[distances[i][0]]]
        return neighbors

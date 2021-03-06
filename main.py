import matplotlib.pyplot as plt
from nn import NN
from ann import ANN
import time
import math


def read_input(f, n):
    f.readline()
    dataset = []
    for i in xrange(n):
        lines = f.readline()[:-1].split()
        x = float(lines[0])
        y = float(lines[1])
        cl = float(lines[2])
        dataset += [[x, y, cl]]
    return dataset


def plot(dataset):
    plt.plot([x[0] for x in dataset if x[2] == 0], [x[1] for x in dataset if x[2] == 0], 'ro')
    plt.plot([x[0] for x in dataset if x[2] == 1], [x[1] for x in dataset if x[2] == 1], 'b^')
    plt.show()


# Read input file
filename = 'data/swissRoll.txt'
train_set = []
with open(filename, 'r') as f:
    for i in xrange(4):
        read_input(f, 10)               # Obsolete test set
        dataset = read_input(f, 10)
        train_set += [dataset]
        dataset = read_input(f, 100)
        train_set += [dataset]
        dataset = read_input(f, 1000)
        train_set += [dataset]

test_set = []
filename = 'data/swissRollTest.txt'
with open(filename, 'r') as f:
    for i in xrange(4):
        dataset = read_input(f, 100)
        test_set += [dataset]

#plot(train_set[2])

# nn_model = NN(train_set=train_set[11], k=3)
# correct = 0
# wrong = 0
# start_time = time.clock()
# for inst in test_set[3]:
#     prediction = nn_model.predict(inst)
#     #print 'prediction:', prediction, 'true class:', inst[-1]
#     if prediction == int(inst[-1]):
#         correct += 1
#     else:
#         wrong += 1
#
# print 'Exact NN'
# print 'correct:', correct
# print 'wrong:', wrong
# print 'time:', time.clock() - start_time


# Test approximate nearest neighbor with locality sensitive hashing
# ann_model = ANN(train_set=train_set[11], k=3, num_buckets=1)
# correct = 0
# wrong = 0
# start_time = time.clock()
# for inst in test_set[3]:
#     prediction = ann_model.predict(inst)
#     #print 'prediction:', prediction, 'true class:', inst[-1]
#     if prediction == int(inst[-1]):
#         correct += 1
#     else:
#         wrong += 1
#
# print 'Approximate NN'
# print 'correct:', correct
# print 'wrong:', wrong
# print 'time:', time.clock() - start_time

#ann_model.plot_buckets(0)
# nn_model.plot_boundary()

def calc_dist(a, b):
    dim = len(a) - 1
    temp_sum = 0
    for i in xrange(dim):
        temp_sum += math.pow(a[i] - b[i], 2)
    return math.sqrt(temp_sum)

filename = 'data/ColorHistogram.asc'
n = 68040
dim = 32
dataset = []
with open(filename, 'r') as f:
    for i in xrange(n):
        vector = [0]*32
        line = f.readline()
        lines = line.split()
        for j in xrange(1, len(lines)):
            vector[j-1] = float(lines[j])
        dataset += [vector]

print 'Experiment effective error COREL'
train_set = dataset[:60000]
test_set = dataset[60001:60100]
k = 1
errors = []
nn_model = NN(train_set=train_set, k=k)
for l in xrange(1, 10):
    temp_sum = 0
    ann_model = ANN(train_set=train_set, k=k, num_buckets=l)
    for instance in test_set:
        start_time = time.clock()
        a = ann_model.get_approximate_neighbors(instance)
        if len(a) < 1:
            continue
        else:
            a = a[0]
        # print 'a:', a
        # print 'time:', time.clock() - start_time
        start_time = time.clock()
        b = nn_model.get_exact_neighbors(instance)[0]
        # print 'b:', b
        # print 'time:', time.clock() - start_time
        dlsh = calc_dist(instance, a)
        dstar = calc_dist(instance, b)
        temp_sum += dlsh / dstar
    error = temp_sum / len(test_set)
    errors += [error]
    print 'bucket:', l, 'error:', error
plt.plot(errors, 'bo-')
plt.show()

# # Experiment 1, effective error
# print 'Experiment effective error'
# errors = []
# for l in xrange(1, 10):
#     temp_sum = 0
#     for instance in test_set[3]:
#         ann_model = ANN(train_set=train_set[11], k=1, num_buckets=l)
#         a = ann_model.get_approximate_neighbors(instance)[0]
#         b = nn_model.get_exact_neighbors(instance)[0]
#         dlsh = calc_dist(instance, a)
#         dstar = calc_dist(instance, b)
#         temp_sum += dlsh / dstar
#     error = temp_sum / len(test_set[3])
#     errors += [error]
#     print 'bucket:', l, 'error:', error
# plt.plot(errors, 'bo-')
# plt.show()

# Experiment 2, miss ratio
print 'Experiment miss ratio'
train_ratio = [3000, 6000, 15000, 30000, 60000]
# train_set = dataset[:3000]
test_set = dataset[60001:68040]
k = 10
miss_ratio = []
for ratio in train_ratio:
    miss = 0
    train_set = dataset[:ratio]
    ann_model = ANN(train_set=train_set, k=k, num_buckets=1)
    for instance in test_set:
        a = ann_model.get_approximate_neighbors(instance)
        # print len(a)
        if len(a) < k:
            miss += 1
            # if calc_effective_erro(a, b)
    miss_ratio += [float(miss) / len(test_set)]
    print 'ratio:', ratio
    print 'miss ratio:', float(miss) / len(test_set)
plt.plot([0.3, 0.6, 1.5, 3, 6], miss_ratio, 'bo-')
plt.show()

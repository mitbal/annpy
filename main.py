import sys
import matplotlib.pyplot as plt
from nn import NN
from ann import ANN
import time

MAX_INT = sys.maxint


def read_input(f, n):
    description = f.readline()
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
test_set = []
train_set = []
with open(filename, 'r') as f:
    for i in xrange(4):
        dataset = read_input(f, 10)
        test_set += [dataset]
        dataset = read_input(f, 10)
        train_set += [dataset]
        dataset = read_input(f, 100)
        train_set += [dataset]
        dataset = read_input(f, 1000)
        train_set += [dataset]

test_set = []
filename = 'data/swissRollTest.txt'
with open(filename, 'r') as f:
    test_set = read_input(f, 100)

#plot(train_set[2])

nn_model = NN(train_set=train_set[11], k=9)
correct = 0
wrong = 0
start_time = time.clock()
for inst in test_set:
    prediction = nn_model.predict(inst)
    #print 'prediction:', prediction, 'true class:', inst[-1]
    if prediction == int(inst[-1]):
        correct += 1
    else:
        wrong += 1

print 'Exact NN'
print 'correct:', correct
print 'wrong:', wrong
print 'time:', time.clock() - start_time


# Test approximate nearest neighbor with locality sensitive hashing
ann_model = ANN(train_set=train_set[11], k=1)
correct = 0
wrong = 0
start_time = time.clock()
for inst in test_set:
    prediction = ann_model.predict(inst)
    #print 'prediction:', prediction, 'true class:', inst[-1]
    if prediction == int(inst[-1]):
        correct += 1
    else:
        wrong += 1

print 'Approximate NN'
print 'correct:', correct
print 'wrong:', wrong
print 'time:', time.clock() - start_time

#ann_model.plot_buckets()

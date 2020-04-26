from operator import itemgetter
from collections import Counter
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


def createDataSet():
    features, labels = load_iris(True)
    index = np.arange(len(features))
    np.random.shuffle(index)

    split = 0.8
    num_train = int(np.floor(len(index) * split))
    index_train = index[:num_train]
    X_train = features[index_train]
    y_train = labels[index_train]
    index_test = index[num_train:]
    X_test = features[index_test]
    y_test = labels[index_test]
    return X_train, y_train, X_test, y_test


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = Counter(labels[sortedDistIndices[:k]])
    return classCount.most_common(1)[0][0]






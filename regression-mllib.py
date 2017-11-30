from pyspark import SparkContext
import numpy as np
from numpy import array
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS

sc = SparkContext ()

def createLabeledPoints(label, points):
    return LabeledPoint(label, points)

studyHours = [
 [ 0, [0.5]],
 [ 0, [0.75]],
 [ 0, [1.0]],
 [ 0, [1.25]],
 [ 0, [1.5]],
 [ 0, [1.75]],
 [ 1, [1.75]],
 [ 0, [2.0]],
 [ 1, [2.25]],
 [ 0, [2.5]],
 [ 1, [2.75]],
 [ 0, [3.0]],
 [ 1, [3.25]],
 [ 0, [3.5]],
 [ 1, [4.0]],
 [ 1, [4.25]],
 [ 1, [4.5]],
 [ 1, [4.75]],
 [ 1, [5.0]],
 [ 1, [5.5]]
]

data = []

for x, y in studyHours:
	data.append(createLabeledPoints(x, y))

model = LogisticRegressionWithLBFGS.train( sc.parallelize(data) )

print (model)

print (model.predict([1]))

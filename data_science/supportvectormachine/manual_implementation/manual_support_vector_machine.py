#!/usr/bin/env python
# coding=utf-8
""" implementing support vector machine from scratch """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")


class SupportVectorMachine:
    def __init__(self, visualization=True):
        self.visualization = visualization

        # these are the colors for the two lines
        self.colors = {1: "r", -1: "b"}

        # unless false is set when the an object is initialized
        # create a new figure and assign this figure to the attribute fig
        # and add the sub plut in the locatio 1,1,1 in for format
        # nrows, ncols, index
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        # self data will store he data that will be passsed through the
        # SVM
        self.data = data

        # { ||w||: [w, b]}
        opt_dict = {}

        # this is the transformation matric for trying all possible
        # conbinations
        transform = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        all_data = []
        # in this loop we are extracting the features into an array
        for yi in self.data:
            print("what yi represents in data")
            print(yi)
            for featureset in self.data[yi]:
                print("what featurset represents in data")
                print(featureset)
                for feature in featureset:
                    print(10 * "#")
                    print("what feature represents in featureset")
                    print(feature)
                    all_data.append(feature)
                    print(
                        "this is the final list of features that were appended"
                    )
                    print(all_data)
        # then we gab the maxminum and minimum value from all features and then
        # we set it to none
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # we create an array with the max feature value and multiply it by
        # .1, .01, and .001. These values represent the the size of the
        # steps that will be taking. The smaller the steps the more times
        # that it will be needed to take
        # support vectors yi(xi,w+b) = 1
        step_sizes = [
            self.max_feature_value * 0.1,
            self.max_feature_value * 0.01,
            # this is the point where it starts to become very
            # expensive
            self.max_feature_value * 0.001,
        ]

        # this one is extremely expensive
        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value * 10
        # for max value in the step size array
        for step in step_sizes:
            # w = an np array with the lates optimum twice
            w = np.array([latest_optimum, latest_optimum])
            # set optimized to equal false
            optimized = False
            # start iterating until optimized gets flipped to false
            while not optimized:
                # arrange is similar to the function range in python. It
                # returns a list created by given parameters arrange
                for b in np.arange(
                    -1 * (self.max_feature_value * b_range_multiple),
                    self.max_feature_value * b_range_multiple,
                    step * b_multiple,
                ):
                    print(10 * "_")
                    print(
                        "This is the np range that we are iterating through "
                    )
                    print(
                        np.arange(
                            -1 * (self.max_feature_value * b_range_multiple),
                            self.max_feature_value * b_range_multiple,
                            step * b_multiple,
                        )
                    )
                    print(b)
                        "This is every element of the array created by ranges"
                    )
                    print(b)
                    for transformation in transform:
                        print("this is each transofrmation in transform array")
                        print(
                            "w_t = array with the latest optimum twice times the transormation array"
                        )
                        w_t = w * transformation
                        print(w_t)
                        found_options = True
                        # weakest link in the SCM findamentally SMO attempts
                        # to fix this a bit
                        # yi(xi.w + b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    print(40 * "#")
                                    print(
                                        "this is the values that has to be greater than one"
                                    )
                                    print(yi * (np.dot(w_t, xi) + b))
                                    found_options = False

                    if found_options:
                        opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print("Optimized a step")
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            # ||w|| : [w, b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

        for i in self.data:
            for xi in self.data:
                yi = i
                print(xi, ":", yi * (np.dot(self.w, xi) + self.b))

    def predict(self, features):
        # sign( x.w+b)
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(
                features[0],
                features[1],
                s=200,
                marker="*",
                c=self.colors[classification],
            )
        return classification

    def visualize(self):
        [
            [
                self.ax.scatter(x[0], x[1], s=100, color=self.colors[i])
                for x in data_dict[i]
            ]
            for i in data_dict
        ]

        # hyper plane = x.w + b
        # v = x.w + b
        # psv = 1
        # psv = -1
        # dec = 0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        datarange = (
            self.min_feature_value * 0.9,
            self.max_feature_value * 1.1,
        )
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]
        # (w.x + b) = 1
        # positive support vector huperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2])

        # (w.x + b) = -1
        # negative support vector huperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2])

        # (w.x + b) = 1
        # positive support vector huperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2])

        plt.show()


data_dict = {
    -1: np.array([[1, 7], [2, 8], [3, 8],]),
    1: np.array([[5, 1], [6, -1], [7, 3],]),
}

svm = SupportVectorMachine()
svm.fit(data=data_dict)

predict_us = [
    [0, 10],
    [1, 3],
    [3, 4],
    [3, 5],
    [5, 5],
    [5, 6],
    [6, -5],
    [5, 8],
    [5, 8],
]

for p in predict_us:
    svm.predict(p)

svm.visualize()

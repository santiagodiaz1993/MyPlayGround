#!/usr/bin/env python
# coding=utf-8
class Model:
    def hello(self):
        print("hello")


class Model2:
    def hello(self):
        print("chao")


example1 = Model()
example2 = Model()
example3 = Model2()
example4 = Model2()

layers = [example1, example2, example3, example4]


for i in layers:
    if i == 2:
        layers[i].prev.hello()
        print(layers[i])
        print(layers[i].next)

        # layers[i].next = layers[i + 1]
    # else:
    #     print(layers[i].prev)
    #     print(layers[i].next)

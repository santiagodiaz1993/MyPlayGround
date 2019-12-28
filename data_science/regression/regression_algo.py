"""Manually finding the best fit line given a range 
of x and y values"""


from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

# choose style
style.use('fivethirtyeight')


xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

def best_fit_slopeand_intercept(xs, ys):
	"""function that returns the slope and b values 
	from an array of x and y values"""
	m = ( ((mean(xs) * mean(ys)) - mean(xs*ys) ) / 
		( (mean(xs)*mean(xs)) - mean(xs*xs)) )
	b = mean(ys) - m * mean(xs) 
	return m, b

m, b = best_fit_slopeand_intercept(xs,ys)	

# this line = for x in xs:
#				regression_line.append((m*x) + b)

regression_line = [(m*x) + b for x in xs]

xs2 = np.array([15, 20, 25])

def predict(xs2):
	"""function that preditcs the y values given the 
	x values"""
	y_values = np.array([])
	for element in xs2:
		y = np.array(element * m + b)
		y_values = np.append(y_values, y) 
	return y_values

ys2 = predict(xs2)

# plotting predicted points pus the best fit line
plt.scatter(xs2, ys2)
plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.plot(xs2, ys2)
plt.show()

""" ** Best fit line **
Manually finding the best fit line given a range 
of x and y values and predicting y values given x
vlaues and graphing them"""


from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

# choose style
style.use('fivethirtyeight')


# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# hm = how much 
# variance = how much difference between data points 
# step = how far on avrage to step up the y value per point
# correlation = positive or negative coorelation
def create_dataset(hm, variance, step=2, correlation=False):
    val = 1 
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

xs, ys = create_dataset(40, 5, 2, correlation='pos')

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



""" ** R Squared Theory ** 
Coefficient of determination is calculated using the squared 
error. We answer the question how good of a fit is the line?
error = distance from point to line squared 

if r**2 is small this means the data is pretty linear. (All 
points are near the line)

SE = squared error
r**2 = coefficient of determination
r**2 = 1 - SE*line(y)/SE*mean(y)

coefficient of determination = confidence of the line
"""


def squared_error(ys_orig, ys_regression_line):
	"""function returns the distance from each y value to the 
	line"""
	return sum((ys_orig - ys_regression_line)**2)

def ceofficient_of_determination(ys_orig, ys_regression_line):
	y_mean_line = [mean(ys_orig) for y in ys_orig]
	square_error_regression_line = squared_error(ys_orig, ys_regression_line) 
	squared_error_y_mean = squared_error(ys_orig, y_mean_line)
	return 1 - (square_error_regression_line/squared_error_y_mean)


r_squared = ceofficient_of_determination(ys, regression_line)

print('The coeficient of determination is')
print(r_squared)





image = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]

flat_image = []

for array in image:
    for element in array:
        flat_image.append(element)


weights = 6 * [0, 0, 0, 0]
print(weights)
print(flat_image)

bias = 4

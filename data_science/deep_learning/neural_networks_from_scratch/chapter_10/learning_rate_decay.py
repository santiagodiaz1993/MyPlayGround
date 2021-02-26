def learning_rate_decay(starting_learning_rate, learning_decay, step):
    learning_rate = starting_learning_rate * (1 / (1 + learning_decay * step))
    return learning_rate


print(learning_rate_decay(1, 0.1, 1))

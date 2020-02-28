import math
import random
import matplotlib.pyplot as plt


class SimpleNeural:
    def __init__(self, learning_rate=0.1):
        self.weights = [random.random() * 4 / 2 for _ in range(3)]
        self.sigmoid_mapper = self.sigmoid
        self.learning_rate = learning_rate

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def predict(self, inputs):
        input_with_polarization = [*inputs, 1]
        weighted_inputs = 0
        for j in range(3):
            weighted_inputs += input_with_polarization[j] * self.weights[j]
        return self.sigmoid_mapper(weighted_inputs)

    def train(self, inputs, expected_predict):
        actual_predict = self.predict(inputs)
        inputs = [*inputs, 1]

        error = actual_predict - expected_predict
        weight_delta = error * actual_predict * (1 - actual_predict)

        for j in range(3):
            self.weights[j] -= weight_delta * inputs[j] * self.learning_rate


def mean_square_error(actual, expected):
    error = 0
    for j in range(len(actual)):
        error += (actual[j] - expected[j]) ** 2
    return error / len(actual)


if __name__ == '__main__':
    train_cases = [
        ([0.26, 0.14], 0),
        ([0.19, 0.45], 1),
        ([0.24, 0.28], 0),
        ([0.03, 0.41], 1),
        ([0.3, 0.42], 0),
        ([-0.06, 0.39], 1),
        ([0.41, 0.57], 0),
        ([-0.26, 0.46], 1),
        ([-0.39, 0.58], 1),
        ([0.52, 0.66], 0),
        ([0.65, 0.74], 0),
        ([0.84, 0.74], 0),
        ([-0.42, 0.76], 1),
        ([-0.24, 0.82], 1),
        ([0.86, 0.53], 0),
        ([-0.04, 0.76], 1),
        ([0.81, 0.43], 0),
        ([0.12, 0.65], 1),
        ([0.72, 0.3], 0),
        ([-0.04, 0.58], 1),
        ([0.58, 0.2], 0),
        ([-0.22, 0.67], 1),
        ([0.42, 0.13], 0),
        ([0.4, 0.28], 0),
        ([0.55, 0.42], 0),
        ([0.7, 0.58], 0),
        ([0.2, 0.5], 1),  # here and below
        ([0.1, 0.4], 1),
        ([0, 0.4], 1),
        ([-0.1, 0.4], 1),
        ([0.1, 0.5], 1),
        ([0, 0.5], 1),
        ([-0.1, 0.5], 1),
        ([-0.2, 0.5], 1),
        ([-0.3, 0.5], 1),
        ([0.1, 0.6], 1),
        ([0, 0.6], 1),
        ([-0.1, 0.6], 1),
        ([-0.2, 0.6], 1),
        ([-0.3, 0.6], 1),
        ([-0.4, 0.6], 1),
        ([0, 0.7], 1),
        ([-0.1, 0.7], 1),
        ([-0.2, 0.7], 1),
        ([-0.3, 0.7], 1),
        ([-0.4, 0.7], 1),
        ([-0.2, 0.8], 1),
        ([-0.3, 0.8], 1),
        ([-0.4, 0.8], 1),
        ([0.3, 0.2], 0),
        ([0.4, 0.2], 0),
        ([0.5, 0.2], 0),
        ([0.6, 0.2], 0),
        ([0.3, 0.3], 0),
        ([0.4, 0.3], 0),
        ([0.5, 0.3], 0),
        ([0.6, 0.3], 0),
        ([0.3, 0.4], 0),
        ([0.4, 0.4], 0),
        ([0.5, 0.4], 0),
        ([0.6, 0.4], 0),
        ([0.7, 0.4], 0),
        ([0.4, 0.5], 0),
        ([0.5, 0.5], 0),
        ([0.6, 0.5], 0),
        ([0.7, 0.5], 0),
        ([0.8, 0.5], 0),
        ([0.5, 0.6], 0),
        ([0.6, 0.6], 0),
        ([0.7, 0.6], 0),
        ([0.6, 0.7], 0),
        ([0.7, 0.7], 0),
        ([0.8, 0.7], 0),
    ]

    epochs = 100
    rate = 0.1

    network = SimpleNeural(learning_rate=rate)

    for i in range(epochs):
        inputs_ = []
        correct_predictions = []
        for input_case, correct_predict in train_cases:
            network.train(input_case, correct_predict)
            inputs_.append(network.predict(input_case))
            correct_predictions.append(correct_predict)

        error = mean_square_error(
            inputs_,
            correct_predictions,
        )

        if i * 1000 / epochs % 100 == 0:
            print(f'On {i * 100 / epochs}% mse = {error}')
        elif i + 1 == epochs:
            print(f'Final mse = {error}')

    count = 0
    for input_stat, correct_predict in train_cases:
        if round(network.predict(input_stat)) != correct_predict:
            print(
                f'For input: {input_stat} the prediction is: '
                f'{network.predict(input_stat)}, expected: {correct_predict}'
            )
            count += 1

    print(f'---\nWeights: {", ".join([str(i) for i in network.weights])}')

    x_s = []
    y_s = []

    for i in range(100):
        for j in range(100):
            prev = 1
            y = network.predict([0.01 * i, 0.01 * j]) > 0.5
            if y != prev:
                x_s.append(0.01 * i)
                y_s.append(0.01 * j)
                prev = y

    fig, (ax) = plt.subplots(
        nrows=1, ncols=1,
        figsize=(14, 10)
    )

    ax.scatter(x=x_s, y=y_s, marker='o', c='#EBAC0C', edgecolor='#FDFDFD')
    ax.set(xlim=(-0.5, 0.9))

    plt.show()

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [1], [1], [0]])  

np.random.seed(42)
weights_input_hidden = np.random.uniform(size=(2, 2))
weights_hidden_output = np.random.uniform(size=(2, 1))

learning_rate = 0.1

for epoch in range(10000):
    hidden_input = np.dot(X, weights_input_hidden)
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output)
    predicted_output = sigmoid(final_input)

    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    hidden_error = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_output = hidden_error * sigmoid_derivative(hidden_output)

    weights_hidden_output += hidden_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_output) * learning_rate

print("Final predicted output:\n", predicted_output)

model = Sequential()
model.add(Dense(2, input_dim=2, activation='sigmoid'))  
model.add(Dense(1, activation='sigmoid'))               

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=10000, verbose=0)

loss, accuracy = model.evaluate(X, y)
print(f"Accuracy: {accuracy * 100:.2f}%")

predictions = model.predict(X)
print("Predictions:\n", predictions)


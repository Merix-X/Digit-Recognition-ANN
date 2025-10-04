from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

# Load and normalize MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data() # This loads 60,000 pre-prepared training images from the tensorflow.keras library and 10,000 test images needed for training and testing the model
x_train, x_test = x_train / 255.0, x_test / 255.0 # This converts the values in x_train and x_test from 0–255 to 0–1 so the model can learn better (Original pixel values range from 0 to 255 (black to white). Neural networks perform better with small input values, so dividing by 255 normalizes pixels between 0 (black) and 1 (white))

# Build the model
"""
Choosing the ideal number of neurons in each layer:
In general:
    - More neurons → the model can be more accurate, but slower and prone to "overfitting".
    - Fewer neurons → the model is faster, but may be less accurate.
Typical approach:
    - Start with a reasonable number (e.g., 64, 128),
    - Test accuracy on test data,
    - If accuracy is low → add neurons or layers,
    - If the model overfits → reduce neurons or add regularization.
"""

model = Sequential([  # Model where layers are stacked in sequence
    Flatten(input_shape=(28, 28)), # Flatten "flattens" the 2D image (28×28) into a 1D array with 784 values => prepares values for the input neurons of the first layer
    Dense(128, activation='relu'), # Hidden layer with 128 neurons, activation relu = if the output is less than 0 → set it to 0, otherwise leave it as is
    Dense(64, activation='relu'), # Another hidden layer with 64 neurons, same activation
    Dense(10, activation='softmax') # Output layer – 10 neurons, each representing one digit (0–9), softmax = "how confident the network is that this digit is the correct one"
])
model.compile(optimizer='adam', # Defines how the model learns – how it adjusts the neuron weights
              loss='sparse_categorical_crossentropy', # Tells how much the model is wrong in its prediction
              metrics=['accuracy']) # Tracks the model accuracy

# Train the model more thoroughly
print("Training the model, please wait...")
model.fit(x_train, # Model trains on the data (x_train
          y_train, # and y_train) for
          epochs=10, # 10 iterations (epochs)
          validation_data=(x_test, y_test), # After each epoch, test with x_test and y_test
          verbose=2) # Controls console output (0 = no output, 1 = progress bar with detailed info, 2 = one line per epoch (shorter, clearer))
model.save("digit_recognition_model.keras")

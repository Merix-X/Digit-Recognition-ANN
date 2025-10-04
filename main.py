import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Function that creates a bar chart based on the model's confidence and individual digits
def bar_chart(values_list, conf_list):
    plt.bar(values_list, conf_list)

    # Labels above each bar
    for i, v in enumerate(conf_list):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center')

    # Labels under each bar
    plt.xticks(values_list, [str(v) for v in values_list])

    plt.title('Confidence Bar Chart')
    plt.xlabel('Digit')
    plt.ylabel('Confidence (%)')
    plt.ylim(0, 110)  # Leave space for the text above the bars
    plt.show()

# Function that processes and recognizes an image with a digit
def predict_from_file():
    path = input("Enter the path to a 28x28 black and white image of a digit: ").strip() # Load the image path (image must be 28x28 pixels with a white background and a black digit)
    try:  # Try/except in case of an error
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # Load the image from the given path
        if img is None:
           raise ValueError("Image not found or cannot be read.")
        img = cv2.resize(img, (28, 28)) # Resize the image to 28x28 pixels (safety check in case it’s larger)
        img = 255 - img  # Invert colors (black digit on white background)
        img = img / 255.0 # Original pixel values are between 0 and 255 (black to white). Neural networks work better with small input values, so we divide by 255 to normalize pixels between 0 (black) and 1 (white)
        img = img.reshape(1, 28, 28) # Prepare the correct shape for the model

        model = tf.keras.models.load_model("digit_recognition_model.keras")

        prediction = model.predict(img) # Load the image, process it, and return 10 numbers with probabilities for each digit
        predicted_digit = np.argmax(prediction) # Get the most likely digit
        confidence = 100 * np.max(prediction) # Get its confidence value

        # Show a table with probability bars for each digit
        conf_list = []
        prediction_array = prediction[0] * 100
        for conf in prediction_array:
            conf_list.append(conf)

        bar_chart([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], conf_list)

        # Print the result
        print(f"\nPredicted digit: {predicted_digit}")
        print(f"Confidence: {confidence:.2f}%") # "confidence" is a number with many decimal places (e.g., 92.345678). ":" marks the beginning of the format command, ".2f" rounds to 2 decimal places and formats as float => 92.345678% becomes 92.35% (much more readable)
        return prediction

    except Exception as e:
        print(f"Error: {e}")

# Main loop that repeats the recognition (prediction) process until the user enters "N"
while True:
    predict_from_file()
    again = input("\nDo you want to test another image? (y/n): ").strip().lower()
    if again != 'y':
        print("Goodbye!")
        break

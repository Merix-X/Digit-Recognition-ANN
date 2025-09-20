import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Funkce vytvářející sloupcový graf na základě jistoty modelu a jednotlivých čísel
def bar_chart(values_list, conf_list):
    plt.bar(values_list, conf_list)

    # Popisky nahoře nad každým sloupcem
    for i, v in enumerate(conf_list):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center')

    # Popisky pod každým sloupcem
    plt.xticks(values_list, [str(v) for v in values_list])

    plt.title('Confidence Bar Chart')
    plt.xlabel('Digit')
    plt.ylabel('Confidence (%)')
    plt.ylim(0, 110)  # Aby bylo místo pro text nad sloupci
    plt.show()

# Funkce která zpracuje a rozpozná obrázek s číslem
def predict_from_file():
    path = input("Enter the path to a 28x28 grayscale image of a digit: ").strip() # Načtení cesty k obrázku (obrázek velký 28x28 pixelů s bílým pozadím a černě nakreslenou číslicí)
    try:  # Pro případ že by došlo k chybě je tu try: except:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # Načtu obrázek podle cesty zadaná v inputu
        if img is None:
           raise ValueError("Image not found or cannot be read.")
        img = cv2.resize(img, (28, 28)) # A převedu velikost obrázku na 28x28 pixelů (spíše takové ujištění kdyby byl obrázek větší)
        img = 255 - img  # Pro jistotu prohodím barvy (černá číslice na bílém pozadí)
        img = img / 255.0 # Původní pixel má hodnotu od 0 do 255 (černá až bílá). Neuronové sítě lépe rozpoznávají a zpracovávají, pokud mají vstupy malá čísla. Proto každý pixel vydělím 255, aby byl v rozmezí 0 (černá) až 1 (bílá)
        img = img.reshape(1, 28, 28) # Připravím správný tvar pro model

        model = tf.keras.models.load_model("digit_recognition_model.keras")

        prediction = model.predict(img) # Načte obrázek, zpracuje ho a vrátí 10 čísel společně s pravděpodobností pro každou číslici
        predicted_digit = np.argmax(prediction) # Nejde nejpravděpodobnější číslo
        confidence = 100 * np.max(prediction) # Zjistí hodnotu

        # Zobrazí tabulku se sloupci pravděpodobnosti pro každé číslo
        conf_list = []
        prediction_array = prediction[0] * 100
        for conf in prediction_array:
            conf_list.append(conf)

        bar_chart([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], conf_list)

        # Vypíše výsledek
        print(f"\nPredicted digit: {predicted_digit}")
        print(f"Confidence: {confidence:.2f}%") # "confidence" je číslo s hodně desetinnými čísly (např. 92.345678). ":" - začátek formátovacího příkazu, ".2f" - zaokrouhlí na 2 desetinná místa a formátuje jako desetinné číslo (f = float) => Výsledkem je z 92.345678% něco jako 92.35% (mnohem čitelnější)
        return prediction

    except Exception as e:
        print(f"Error: {e}")

# Hlavní smyčka která celý proces rozpoznávání (predikce) opakuje dokud uživatel nezadá "N"
while True:
    predict_from_file()
    again = input("\nDo you want to test another image? (y/n): ").strip().lower()
    if again != 'y':
        print("Goodbye!")
        break
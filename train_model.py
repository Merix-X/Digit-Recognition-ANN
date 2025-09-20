from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist

# Load and normalize MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data() # Tohle načte 60 000 předem připravených trénovacích obrázků obsažených v knihovně tensorflow.keras a 10 000 testovacích obrázků potřebných pro trénování a testování modelu
x_train, x_test = x_train / 255.0, x_test / 255.0 # Tohle zase převede hodnoty v x_train, x_test z 0-255 na 0-1 aby se model lépe učil (Původní pixel má hodnotu od 0 do 255 (černá až bílá). Neuronové sítě se učí lépe, pokud mají vstupy malá čísla. Proto každý pixel vydělím 255, aby byl v rozmezí 0 (černá) až 1 (bílá))

# Build the model
"""
Pro výběr ideálního počtu neuronů v každé vrstvě:
Obecně platí:
    - Více neuronů → model může být přesnější, ale pomalejší a může se "přeučit" (overfitting).
    - Méně neuronů → model je rychlejší, ale může být nepřesný.
Obvyklý postup:
    - Začít s rozumným počtem (třeba 64, 128),
    - Otestovat přesnost na testovacích datech,
    - Pokud je přesnost nízká → přidat neurony nebo vrstvy,
    - Pokud se model přeučuje → ubrat neurony nebo přidat regularizaci.
"""

model = Sequential([  # Model kde jdou vrstvy po sobě
    Flatten(input_shape=(28, 28)), # Flatten v podstatě "zploští" 2D obrázek s číslem 28×28 na 1D pole s 784 hodnotami => Připravý hodnoty pro jednotlivé vstupní neurony první vrstvy
    Dense(128, activation='relu'), # Skrytá vrstva s 128 neurony, aktivace relu = pokud je výstup menší než 0 → nastaví 0, ostatní ponechá
    Dense(64, activation='relu'), # Další skrytá vrstva s 64 neurony, ta samá aktivace
    Dense(10, activation='softmax') # Výstupní vrstva – 10 neuronů, každý reprezentuje jednu číslici (0–9), softmax = "jak moc si síť myslí, že je to právě tahle číslice"
])
model.compile(optimizer='adam', # způsob, jak se model "učí" – jak model ladí jednotlivé váhy neuronů
              loss='sparse_categorical_crossentropy', # Říká jak moc se model v daném rozpoznání mýlí
              metrics=['accuracy']) # Sleduje přesnost modelu

# Train the model more thoroughly
print("Training the model, please wait...")
model.fit(x_train, # Model se učí na datech (x_train
          y_train, # a y_train) po dobu
          epochs=10, # 10 opakování (epoch)
          validation_data=(x_test, y_test), # Po každé epoše se otestuje pomocí x_test a y_test
          verbose=2) # Na základě hodnoty (tady 2) nastavuje výstup do konzole (0 = žádný výstup, 1 = progress bar a hodně dalších informací, 2 = jen jeden řádek výpisu (stručnější, přehlednější))
model.save("digit_recognition_model.keras")
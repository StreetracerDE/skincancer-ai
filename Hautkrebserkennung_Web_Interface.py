import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Bildgröße wie beim Training
IMAGE_SIZE = 150

# Modell laden (GANZES Modell mit Architektur)
# st.cache_resource, um das Modell nur einmal zu laden
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Hautkrebserkennung_best_model.keras")

model = load_model()

# Streamlit UI
st.title("🩺 Hautkrebserkennung mit KI")
st.write("Diese Anwendung erkennt potenziellen Hautkrebs auf Basis eines Bildes.")

uploaded_file = st.file_uploader("Wähle ein Hautbild aus (JPG oder PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    # Geändert von use_column_width auf use_container_width, um die Veraltungs-Warnung zu beheben
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    # Bild vorbereiten
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(image) / 255.0  # Normalisierung
    image_array = np.expand_dims(image_array, axis=0)  # Batch dimension

    # Vorhersage
    prediction = model.predict(image_array)[0][0]  # Einzelwert extrahieren

    # Ergebnis anzeigen
    st.subheader("🔍 Vorhersageergebnis")
    st.write(f"Wahrscheinlichkeit für **malignen Hautkrebs**: {prediction:.2%}")
    st.write(f"Wahrscheinlichkeit für **gutartigen Befund**: {(1 - prediction):.2%}")

    if prediction > 0.5:
        st.error("⚠️ Achtung: Verdacht auf malignen Hautkrebs. Eine ärztliche Untersuchung wird empfohlen.")
    else:
        st.success("✅ Kein Hinweis auf malignen Hautkrebs. Trotzdem regelmäßige Kontrolle empfohlen.")

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Bildgröße wie beim Training
IMAGE_SIZE = 64

# Modell laden (GANZES Modell mit Architektur)
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
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    # Bild vorbereiten
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(image) / 255.0  # Normalisierung
    image_array = np.expand_dims(image_array, axis=0)  # Batch dimension

    # Vorhersage
    prediction = model.predict(image_array)[0][0]  # Einzelwert extrahieren

    # Ergebnis anzeigen
    st.subheader("🔍 Vorhersageergebnis")
    
    # Entscheidung treffen, welche Wahrscheinlichkeit als "Sicherheit" angezeigt wird
    if prediction > 0.5:
        # Modell ist sicherer, dass es maligne ist
        confidence = prediction
        result_text = "maligner Hautkrebs"
        st.error("⚠️ Achtung: Verdacht auf malignen Hautkrebs. Eine ärztliche Untersuchung wird empfohlen.")
    else:
        # Modell ist sicherer, dass es gutartig ist
        confidence = 1 - prediction
        result_text = "gutartiger Befund"
        st.success("✅ Kein Hinweis auf malignen Hautkrebs. Trotzdem regelmäßige Kontrolle empfohlen.")
    
    # Zeigt nur die höchste Wahrscheinlichkeit (Sicherheit) an
    st.write(f"Das Modell ist sich zu **{confidence:.2%}** sicher, dass es sich um einen **{result_text}** handelt.")

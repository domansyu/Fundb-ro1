import streamlit as st
import os
import sqlite3
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
from datetime import datetime
# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="Schul-Fundbüro", layout="wide")

st.title("Digitales Schul-Fundbüro")

# Sidebar Navigation
menu = st.sidebar.selectbox(
    "Navigation",
    ["Finder (Upload)", "Verloren & Suchen", "Admin"]

# ==============================
# Konfiguration
# ==============================

MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"
IMAGE_FOLDER = "images"
DB_PATH = "fundbuero.db"
ADMIN_PASSWORD = "admin123"  # Hier kannst du das Passwort ändern

# ==============================
# Initialisierung
# ==============================

# Ordner erstellen
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Datenbank initialisieren
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            category TEXT,
            confidence REAL,
            upload_date TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ==============================
# Modell laden
# ==============================

@st.cache_resource
def load_ai_model():
    try:
        model = load_model(MODEL_PATH, compile=False)
        class_names = open(LABELS_PATH, "r").readlines()
        return model, class_names
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return None, None

model, class_names = load_ai_model()

# ==============================
# KI-Vorhersage
# ==============================

def predict_image(image):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])

    return class_name, confidence_score

# ==============================
# Datenbankfunktionen
# ==============================

def insert_item(filename, category, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO items (filename, category, confidence, upload_date)
        VALUES (?, ?, ?, ?)
    """, (filename, category, confidence, datetime.now().strftime("%d.%m.%Y %H:%M")))
    conn.commit()
    conn.close()

def get_items_by_category(category):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM items WHERE category=?", (category,))
    items = c.fetchall()
    conn.close()
    return items

def get_all_items():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM items")
    items = c.fetchall()
    conn.close()
    return items

def delete_item(item_id, filename):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM items WHERE id=?", (item_id,))
    conn.commit()
    conn.close()

    image_path = os.path.join(IMAGE_FOLDER, filename)
    if os.path.exists(image_path):
        os.remove(image_path)

def get_total_count():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM items")
    count = c.fetchone()[0]
    conn.close()
    return count


)

# ==============================
# 1. Upload-Seite
# ==============================

if menu == "Finder (Upload)":
    st.header("Gegenstand hochladen")

    uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and model is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Hochgeladenes Bild", width=300)

        if st.button("Klassifizieren und speichern"):
            with st.spinner("KI analysiert das Bild..."):
                category, confidence = predict_image(image)

            confidence_percent = round(confidence * 100, 2)

            # Eindeutiger Dateiname
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}.jpg"
            image.save(os.path.join(IMAGE_FOLDER, filename))

            insert_item(filename, category, confidence_percent)

            st.success("Gegenstand erfolgreich gespeichert!")
            st.write(f"**Kategorie:** {category}")
            st.write(f"**Konfidenz:** {confidence_percent}%")

# ==============================
# 2. Such-Seite
# ==============================

elif menu == "Verloren & Suchen":
    st.header("Nach Gegenständen suchen")

    st.write(f"Gesamtzahl gespeicherter Gegenstände: **{get_total_count()}**")

    categories = [name.strip() for name in class_names] if class_names else []
    selected_category = st.selectbox("Kategorie auswählen", categories)

    if selected_category:
        items = get_items_by_category(selected_category)

        if items:
            for item in items:
                st.image(
                    os.path.join(IMAGE_FOLDER, item[1]),
                    caption=f"{item[2]} | {item[3]}% | {item[4]}",
                    width=300
                )
        else:
            st.info("Keine Gegenstände in dieser Kategorie gefunden.")

# ==============================
# 3. Admin-Seite
# ==============================

elif menu == "Admin":
    st.header("Admin-Bereich")

    password = st.text_input("Passwort eingeben", type="password")

    if password == ADMIN_PASSWORD:
        st.success("Zugriff gewährt")

        items = get_all_items()

        for item in items:
            col1, col2 = st.columns([3, 1])

            with col1:
                st.image(
                    os.path.join(IMAGE_FOLDER, item[1]),
                    caption=f"{item[2]} | {item[3]}% | {item[4]}",
                    width=250
                )

            with col2:
                if st.button(f"Löschen ID {item[0]}"):
                    delete_item(item[0], item[1])
                    st.warning("Eintrag gelöscht.")
                    st.rerun()

    elif password != "":
        st.error("Falsches Passwort.")

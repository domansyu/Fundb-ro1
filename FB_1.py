import streamlit as st
import os
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
from datetime import datetime
from supabase import create_client

# ==============================
# Streamlit Setup
# ==============================

st.set_page_config(page_title="Schul-Fundbüro", layout="wide")

# ==============================
# Supabase Konfiguration
# ==============================

SUPABASE_URL = "DEINE_URL"
SUPABASE_KEY = "DEIN_ANON_KEY"
BUCKET_NAME = "item-images"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==============================
# KI Modell
# ==============================

MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"
ADMIN_PASSWORD = "admin123"

# ==============================
# Modell laden
# ==============================

@st.cache_resource
def load_ai_model():
    model = load_model(MODEL_PATH, compile=False)
    class_names = open(LABELS_PATH, "r").readlines()
    return model, class_names

model, class_names = load_ai_model()

# ==============================
# KI Vorhersage
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
# Supabase Funktionen
# ==============================

def insert_item(filename, category, confidence):

    supabase.table("items").insert({
        "filename": filename,
        "category": category,
        "confidence": confidence,
        "upload_date": datetime.now().strftime("%d.%m.%Y %H:%M")
    }).execute()


def get_items_by_category(category):

    data = supabase.table("items") \
        .select("*") \
        .eq("category", category) \
        .execute()

    return data.data


def get_all_items():

    data = supabase.table("items") \
        .select("*") \
        .execute()

    return data.data


def delete_item(item_id, filename):

    supabase.table("items") \
        .delete() \
        .eq("id", item_id) \
        .execute()

    supabase.storage.from_(BUCKET_NAME).remove([filename])


def get_total_count():

    data = supabase.table("items") \
        .select("id") \
        .execute()

    return len(data.data)

# ==============================
# Streamlit UI
# ==============================

st.title("Digitales Schul-Fundbüro")

menu = st.sidebar.selectbox(
    "Navigation",
    ["Finder (Upload)", "Verloren & Suchen", "Admin"]
)

# ==============================
# Upload
# ==============================

if menu == "Finder (Upload)":

    st.header("Gegenstand hochladen")

    uploaded_file = st.file_uploader(
        "Bild hochladen", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=300)

        if st.button("Klassifizieren und speichern"):

            with st.spinner("KI analysiert Bild..."):

                category, confidence = predict_image(image)

            confidence_percent = round(confidence * 100, 2)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}.jpg"

            # Bild in Bytes konvertieren
            from io import BytesIO
            buffer = BytesIO()
            image.save(buffer, format="JPEG")

            # Upload zu Supabase Storage
            supabase.storage.from_(BUCKET_NAME).upload(
                filename,
                buffer.getvalue()
            )

            insert_item(filename, category, confidence_percent)

            st.success("Gegenstand gespeichert!")

            st.write(f"Kategorie: {category}")
            st.write(f"Konfidenz: {confidence_percent}%")

# ==============================
# Suche
# ==============================

elif menu == "Verloren & Suchen":

    st.header("Nach Gegenständen suchen")

    st.write(
        f"Gesamtzahl gespeicherter Gegenstände: **{get_total_count()}**"
    )

    categories = [name.strip() for name in class_names]
    selected_category = st.selectbox("Kategorie auswählen", categories)

    if selected_category:

        items = get_items_by_category(selected_category)

        if items:

            for item in items:

                image_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{item['filename']}"

                st.image(
                    image_url,
                    caption=f"{item['category']} | {item['confidence']}% | {item['upload_date']}",
                    width=300
                )

        else:

            st.info("Keine Gegenstände gefunden.")

# ==============================
# Admin
# ==============================

elif menu == "Admin":

    st.header("Admin-Bereich")

    password = st.text_input("Passwort", type="password")

    if password == ADMIN_PASSWORD:

        items = get_all_items()

        for item in items:

            col1, col2 = st.columns([3,1])

            image_url = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{item['filename']}"

            with col1:

                st.image(
                    image_url,
                    caption=f"{item['category']} | {item['confidence']}% | {item['upload_date']}",
                    width=250
                )

            with col2:

                if st.button(f"Löschen {item['id']}"):

                    delete_item(item["id"], item["filename"])

                    st.warning("Eintrag gelöscht")
                    st.rerun()

    elif password != "":
        st.error("Falsches Passwort.")


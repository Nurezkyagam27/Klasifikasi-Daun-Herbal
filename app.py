import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- Label dan Khasiat ---
class_names = [
    'Belimbing Wuluh', 'Jambu Biji', 'Jeruk Nipis', 'Kemangi',
    'Lidah Buaya', 'Nangka', 'Pandan', 'Pepaya', 'Seledri', 'Sirih'
]

khasiat_daun = {
    'Belimbing Wuluh': 'Menurunkan tekanan darah, mengatasi batuk, dan digunakan sebagai antiseptik alami.',
    'Jambu Biji': 'Mengatasi diare, mempercepat penyembuhan luka, dan menjaga kesehatan jantung.',
    'Jeruk Nipis': 'Meningkatkan daya tahan tubuh, melancarkan pencernaan, dan membantu detoksifikasi.',
    'Kemangi': 'Menambah nafsu makan, memiliki sifat antibakteri, dan meredakan stres.',
    'Lidah Buaya': 'Menyembuhkan luka bakar, melembapkan kulit, dan menutrisi rambut.',
    'Nangka': 'Sumber energi, mendukung pencernaan, dan mengandung antioksidan alami.',
    'Pandan': 'Menambah aroma masakan, menurunkan tekanan darah, dan sebagai penenang alami.',
    'Pepaya': 'Melancarkan pencernaan, meningkatkan trombosit, dan bersifat anti-kanker.',
    'Seledri': 'Menurunkan tekanan darah, membantu detoksifikasi, dan mengandung antioksidan.',
    'Sirih': 'Antiseptik alami, menghentikan pendarahan ringan, dan menjaga kesehatan mulut.'
}

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mobilenetv2_model.h5')

# --- Preprocessing ---
def preprocess_and_predict(model, image):
    image = image.convert('RGB').resize((224, 224))
    image_array = preprocess_input(np.expand_dims(np.array(image), axis=0))
    prediction = model.predict(image_array)
    idx = np.argmax(prediction)
    return class_names[idx], np.max(prediction) * 100

# --- UI Layout ---
st.set_page_config(page_title="Klasifikasi Daun Herbal", layout="centered")
st.markdown("<h3 style='text-align: center;'><span style='color:#0abf53;'>CARI</span> TANAMAN HERBAL<br><span style='color:#1abc9c;'>DENGAN</span> SATU KLIK!!</h3>", unsafe_allow_html=True)
st.markdown("---")

# PILIHAN INPUT GAMBAR
st.subheader("Ambil atau Unggah Gambar Daun")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload gambar dari file", type=["jpg", "jpeg", "png"])
with col2:
    camera_image = st.camera_input("Ambil gambar dari kamera")

model = load_model()
image = None

if uploaded_file:
    image = Image.open(uploaded_file)
elif camera_image:
    image = Image.open(camera_image)

if image:
    st.image(image, caption="Gambar Daun yang Diunggah", use_column_width=True)
    with st.spinner("Menganalisis gambar..."):
        predicted_class, confidence = preprocess_and_predict(model, image)
        manfaat = khasiat_daun.get(predicted_class, "Khasiat tidak ditemukan.")

    st.markdown("---")
    st.markdown(f"<p style='text-align:center; font-size:24px; color:#0abf53;'><strong>{predicted_class}</strong></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; font-size:16px;'>Tingkat Keyakinan: <strong>{confidence:.2f}%</strong></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; font-size:18px; color:#444; border-left: 4px solid #0abf53; padding-left: 10px;'><strong>Khasiat:</strong> {manfaat}</p>", unsafe_allow_html=True)
else:
    st.info("Silakan unggah atau ambil gambar daun terlebih dahulu.")

# --- Footer Ikon ---
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)
with col2:
    st.markdown('<div style="text-align:center;"><img src="https://img.icons8.com/ios/50/00e676/camera--v1.png" width="40"/></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div style="text-align:center;"><img src="https://img.icons8.com/ios/50/00e676/image.png" width="40"/></div>', unsafe_allow_html=True)

import cv2
import streamlit as st
import numpy as np
from PIL import Image

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Araç Tespit Uygulaması",
    page_icon="🚗",
    layout="wide"
)

# CSS ile stil
st.markdown("""
<style>
.main-container {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
}
.title {
    color: #2c3e50;
    text-align: center;
    font-weight: bold;
}
.upload-box {
    background-color: #ffffff;
    border: 2px dashed #3498db;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}
.upload-box:hover {
    background-color: #f1f8ff;
    border-color: #2980b9;
}
.info-box {
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    color: #2c3e50;
    background-color: #ecf0f1;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Cascade Classifier'ları yükle
car_cascade_src = 'C:\\Users\\furka\\PycharmProjects\\HelloWorld\\detect-and-count-vehicles\\cars.xml'  # Cascade xml dosyanızın yolu
car_cascade = cv2.CascadeClassifier(car_cascade_src)


def process_image(image_file):
    # Fotoğrafı aç
    image = Image.open(image_file)
    image_arr = np.array(image)

    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)

    # Arabaları tespit et
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # Her araç için dikdörtgen çiz
    for (x, y, w, h) in cars:
        cv2.rectangle(image_arr, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Toplam araç sayısını hesapla
    total_cars = len(cars)

    return Image.fromarray(image_arr), total_cars


def main():
    # Ana konteyner
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Başlık
    st.markdown('<h1 class="title">🚗 Akıllı Araç Tespit Uygulaması</h1>', unsafe_allow_html=True)

    # Açıklama
    st.markdown("""
    <p style="text-align: center; color: #7f8c8d;">
    Fotoğrafınızı yükleyin ve araçları otomatik olarak tespit edin!
    </p>
    """, unsafe_allow_html=True)

    # Fotoğraf yükleme
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    image_file = st.file_uploader(
        "Bir fotoğraf yükleyin",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if image_file is not None:
        # Sütunları oluştur
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Yüklenen Fotoğraf")
            st.image(image_file, use_column_width=True)

        with col2:
            # Fotoğrafı işle
            st.markdown('<div class="info-box">Fotoğraf işleniyor...</div>', unsafe_allow_html=True)
            result_image, car_count = process_image(image_file)

            st.subheader("Tespit Edilen Fotoğraf")
            st.image(result_image, use_column_width=True)

        # Araç sayısını göster
        st.success(f"🚘 Toplam Tespit Edilen Araç Sayısı: {car_count}")

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

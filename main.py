import streamlit as st
from PIL import Image
import uuid
import pandas as pd
import numpy as np
import cv2
from faceplusplus import analyze_face
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch
import os

# ---------------------------- Page Config ---------------------------- #
st.set_page_config(page_title="Skincare Recommendation System", layout="centered")

st.markdown(
    "<h3 style='text-align: center;'>🧴 Skincare Recommendation System</h3>",
    unsafe_allow_html=True
)

# ---------------------------- Load Hugging Face Model ---------------------------- #
@st.cache_resource
def load_hf_model():
    model = ViTForImageClassification.from_pretrained('dima806/skin_types_image_detection')
    extractor = ViTFeatureExtractor.from_pretrained('dima806/skin_types_image_detection')
    return model, extractor

hf_model, hf_extractor = load_hf_model()

@st.cache_data
def load_recommendations():
    df = pd.read_excel("recommendation.xlsx")
    return df

recommendation_df = load_recommendations()

# ---------------------------- Explain Score ---------------------------- #
def explain_score(name, value):
    if name == "health":
        if value >= 4: return "Excellent skin health"
        elif value >= 2: return "Moderate skin health"
        else: return "Needs care – low skin health"
    elif name == "stain":
        if value < 5: return "No visible stains"
        elif value < 15: return "Few visible stains"
        else: return "Prominent stains on skin"
    elif name == "dark_circle":
        if value < 3: return "No dark circles"
        elif value < 5: return "Mild dark circles"
        elif value < 7: return "Moderate dark circles"
        elif value < 9: return "Heavy dark circles"
        else: return "Prominent dark circles"
    elif name == "acne":
        if value < 2: return "Clear skin"
        elif value < 5: return "Occasional acne"
        else: return "Acne-prone skin"
    return ""

# ---------------------------- Dark Circle Detection ---------------------------- #
def detect_dark_circles_opencv(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    dark_circle_score = 0
    for (x, y, w, h) in faces:
        roi_under_eye = gray[y + int(h * 0.5):y + h, x:x + w]
        if roi_under_eye.size == 0:
            continue
        mean_intensity = np.mean(roi_under_eye)
        darkness = max(0, 255 - mean_intensity)
        dark_circle_score = min(10, darkness / 25)

    return round(dark_circle_score, 2)

# ---------------------------- Face Preprocessing ---------------------------- #
def preprocess_image(image):
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        face_roi = img[y:y + h, x:x + w]
    else:
        st.warning("⚠️ No clear face detected. Using full image.")
        face_roi = img

    resized = cv2.resize(face_roi, (224, 224))
    rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    gray_eq = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray_eq)
    final_img = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(final_img)

# ---------------------------- Image Enhancement & Quality Check ---------------------------- #
def enhance_image(img_pil):
    img = np.array(img_pil.convert("RGB"))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_eq)

def is_low_quality(img_pil):
    gray = np.array(img_pil.convert("L"))
    brightness = np.mean(gray)
    return brightness < 70 or brightness > 200

# ---------------------------- Skin Type Detection ---------------------------- #
def detect_skin_type_huggingface(image):
    img_rgb = image.convert("RGB")
    inputs = hf_extractor(images=img_rgb, return_tensors="pt")
    outputs = hf_model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().detach().numpy()

    id2label = hf_model.config.id2label
    skin_types = [id2label[i] for i in range(len(probs))]
    confidence_scores = [round(float(p) * 100, 2) for p in probs]

    df = pd.DataFrame({
        "Skin Type": skin_types,
        "Confidence (%)": confidence_scores
    }).sort_values(by="Confidence (%)", ascending=False).reset_index(drop=True)

    final_prediction = df.iloc[0]["Skin Type"]
    confidence = df.iloc[0]["Confidence (%)"]
    return final_prediction, confidence, df

# ---------------------------- Recommendations ---------------------------- #
def get_recommendations(skin_type):
    return recommendation_df[recommendation_df['Skin Type'].str.lower() == skin_type.lower()]

def show_recommendations(skin_type):
    st.subheader("🧴 Product Recommendations")
    filtered_df = get_recommendations(skin_type)

    if not filtered_df.empty:
        for category in ["Face Wash", "Moisturizer", "Serum", "Sunscreen"]:
            products = filtered_df[filtered_df["Category"] == category]
            if not products.empty:
                with st.expander(f"📦 {category}s ({len(products)})"):
                    for _, row in products.iterrows():
                        st.image(row["Image URL"], width=150)
                        st.markdown(f"**{row['Product Name']}**")
                        st.markdown(f"[🛜️ Buy Now]({row['Buy Link']})", unsafe_allow_html=True)
                        st.markdown("---")
    else:
        st.warning("⚠️ No products found for this skin type.")

# ---------------------------- Analysis Logic ---------------------------- #
def analyze_flow(image_pil):
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    image_pil.save(temp_filename)

    result = analyze_face(temp_filename)

    if "error_message" in result:
        st.error(result["error_message"])
        return

    if "faces" not in result or len(result["faces"]) == 0:
        st.error("😕 Could not detect face using Face++. Try a clearer or frontal photo.")
        return

    skinstatus = result["faces"][0]["attributes"]["skinstatus"]
    skinstatus["dark_circle"] = detect_dark_circles_opencv(temp_filename)

    st.subheader("📊 Face++ Analysis")
    for key, val in skinstatus.items():
        note = f"_via OpenCV_" if key == "dark_circle" else ""
        st.markdown(f"**{key.capitalize()}**: {val:.2f} {note} → _{explain_score(key, val)}_")

    preprocessed_image = preprocess_image(image_pil)
    skin_type, confidence, df = detect_skin_type_huggingface(preprocessed_image)

    if confidence < 40:
        st.warning(f"⚠️ Low confidence in prediction ({confidence}%). Try a clearer photo.")
    st.success(f"🎯 Skin Type: **{skin_type}** ({confidence}%)")
    st.dataframe(df)

    st.session_state["skin_type"] = skin_type  # 👈 Save for use in other pages


    show_recommendations(skin_type)
    os.remove(temp_filename)

     # Show the page link only after skin_type is available
    st.page_link("pages/skincare_actives.py", label="➡️ Go to Skincare Actives Page", icon="🧴")

# ---------------------------- Main Section ---------------------------- #
st.subheader("📸 Take or Upload a Photo")
input_method = st.radio("Select Input Method", ("Upload Image", "Use Webcam"))

image_pil = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Upload a clear face image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Uploaded Image", use_container_width=True)

elif input_method == "Use Webcam":
    picture = st.camera_input("Take a picture")
    if picture:
        image_pil = Image.open(picture)
        image_pil = enhance_image(image_pil)

        if is_low_quality(image_pil):
            st.warning("⚠️ Lighting may be too dark or bright. Try capturing under natural lighting or adjust your camera angle.")

        st.image(image_pil, caption="Enhanced Webcam Image", use_container_width=True)

if image_pil is not None:
    if st.button("🧪 Analyze Skin"):
        with st.spinner("Analyzing..."):
            analyze_flow(image_pil)

            

# ---------------------------- Custom Background Color ---------------------------- #
st.markdown("""
    <style>
        body {
            background-color: #fce4ec;
        }
        .stApp {
            background-color: #fce4ec;
        }
    </style>
""", unsafe_allow_html=True)

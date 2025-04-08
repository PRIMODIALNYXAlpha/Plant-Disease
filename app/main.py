import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st




# Detailed disease information dictionary
disease_info = {
    'Apple___Apple_scab': {
        "name": "Apple Scab",
        "symptoms": "Dark, scaly lesions on leaves and fruit.",
        "causes": "Caused by the fungus *Venturia inaequalis*.",
        "prevention": "Plant resistant apple varieties and rake fallen leaves to reduce infection sources.",
        "treatment": "Apply appropriate fungicides during early leaf development stages."
    },
    'Apple___Black_rot': {
        "name": "Black Rot in Apples",
        "symptoms": "Dark, sunken lesions on fruit; leaves may show brown spots.",
        "causes": "Fungal infection by *Botryosphaeria obtusa*.",
        "prevention": "Prune out dead wood and remove mummified fruits to reduce fungal spores.",
        "treatment": "Use recommended fungicides during the growing season."
    },
    'Apple___Cedar_apple_rust': {
        "name": "Cedar Apple Rust",
        "symptoms": "Bright orange spots on leaves; can cause premature leaf drop.",
        "causes": "Fungus *Gymnosporangium juniperi-virginianae*, which requires both apple and cedar trees to complete its lifecycle.",
        "prevention": "Remove nearby cedar trees if possible and choose resistant apple varieties.",
        "treatment": "Apply fungicides as a preventive measure during spring."
    },
    'Apple___healthy': {
        "name": "Healthy Apple",
        "symptoms": "No visible disease symptoms.",
        "causes": "N/A",
        "prevention": "Maintain proper orchard hygiene and monitor regularly.",
        "treatment": "Continue regular care and inspection."
    },
    'Blueberry___healthy': {
        "name": "Healthy Blueberry",
        "symptoms": "No signs of disease; vibrant green leaves and healthy fruit.",
        "causes": "N/A",
        "prevention": "Ensure acidic soil with good drainage and proper sunlight.",
        "treatment": "Regular pruning and monitoring for pests."
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        "name": "Powdery Mildew on Cherry",
        "symptoms": "White, powdery fungal growth on leaves and fruit.",
        "causes": "Caused by the fungus *Podosphaera clandestina*.",
        "prevention": "Prune trees to improve air circulation and reduce humidity.",
        "treatment": "Apply sulfur-based fungicides promptly when symptoms are first noticed."
    },
    'Cherry_(including_sour)___healthy': {
        "name": "Healthy Cherry",
        "symptoms": "No disease symptoms; healthy foliage and fruit.",
        "causes": "N/A",
        "prevention": "Regular pruning and proper fertilization.",
        "treatment": "Continue standard care practices."
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        "name": "Gray Leaf Spot in Corn",
        "symptoms": "Grayish, rectangular lesions on leaves.",
        "causes": "Fungus *Cercospora zeae-maydis*.",
        "prevention": "Rotate crops and select resistant hybrids.",
        "treatment": "Apply fungicides at the onset of symptoms."
    },
    'Corn_(maize)___Common_rust_': {
        "name": "Common Rust in Corn",
        "symptoms": "Reddish-brown pustules on leaves.",
        "causes": "Fungus *Puccinia sorghi*.",
        "prevention": "Plant resistant corn varieties.",
        "treatment": "Use fungicides if infection is severe."
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        "name": "Northern Leaf Blight in Corn",
        "symptoms": "Long, cigar-shaped gray-green lesions on leaves.",
        "causes": "Fungus *Exserohilum turcicum*.",
        "prevention": "Choose resistant hybrids and practice crop rotation.",
        "treatment": "Apply appropriate fungicides when necessary."
    },
    'Corn_(maize)___healthy': {
        "name": "Healthy Corn",
        "symptoms": "No visible disease; robust growth.",
        "causes": "N/A",
        "prevention": "Regular monitoring and balanced fertilization.",
        "treatment": "Maintain proper field sanitation."
    },
    'Grape___Black_rot': {
        "name": "Black Rot in Grapes",
        "symptoms": "Circular brown lesions on leaves; shriveled black fruit.",
        "causes": "Fungus *Guignardia bidwellii*.",
        "prevention": "Remove mummified berries and prune infected vines.",
        "treatment": "Apply fungicides early in the growing season."
    },
    'Grape___Esca_(Black_Measles)': {
        "name": "Esca (Black Measles) in Grapes",
        "symptoms": "Interveinal leaf 'striping' and spotted, discolored fruit.",
        "causes": "Complex of fungal pathogens.",
        "prevention": "Avoid vine injuries and manage irrigation properly.",
        "treatment": "No effective cure; remove and destroy affected vines."
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        "name": "Leaf Blight in Grapes",
        "symptoms": "Small dark spots with yellow halos on leaves.",
        "causes": "Fungus *Pseudocercospora vitis*.",
        "prevention": "Ensure good air circulation and remove affected leaves.",
        "treatment": "Apply recommended fungicides during early symptom development."
    },
    'Grape___healthy': {
        "name": "Healthy Grape",
        "symptoms": "No disease symptoms; healthy leaves and fruit.",
        "causes": "N/A",
        "prevention": "Regular pruning and disease monitoring.",
        "treatment": "Continue with standard vineyard management practices."
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        "name": "Citrus Greening (Huanglongbing)",
        "symptoms": "Yellowing of leaves, misshapen and bitter fruit.",
        "causes": "Bacterial infection spread by the Asian citrus psyllid.",
        "prevention": "Control psyllid populations and remove infected trees.",
        "treatment": "No cure available; focus on prevention and control measures."
    },
    'Peach___Bacterial_spot': {
        "name": "Bacterial Spot in Peach",
        "symptoms": "Small, dark lesions on leaves and fruit.",
        "causes": "Bacterial infection by *Xanthomonas arboricola*.",
        "prevention": "Use resistant varieties and avoid overhead irrigation.",
        "treatment": "Apply copper-based bactericides during early spring."
    },
    'Peach___healthy': {
        "name": "Healthy Peach",
        "symptoms": "No visible disease; healthy leaves and fruit.",
        "causes": "N/A",
        "prevention": "Regular monitoring and proper care.",
        "treatment": "Continue standard orchard management."
    },
    'Pepper,_bell___Bacterial_spot': {
        "name": "Bacterial Spot in Pepper",
        "symptoms": "Small, dark lesions on leaves and fruit.",
        "causes": "Bacterial infection by *Xanthomonas campestris*.",
        "prevention": "Use resistant varieties and avoid overhead irrigation.",
        "treatment": "Apply copper-based bactericides during early spring."
    },
    'Pepper,_bell___healthy': {
        "name": "Healthy Bell Pepper",
        "symptoms": "No disease symptoms; vibrant green leaves.",
        "causes": "N/A",
        "prevention": "Regular monitoring and proper care.",
        "treatment": "Continue standard care practices."
    },
    'Potato___Early_blight': {
        "name": "Early Blight in Potato",
        "symptoms": "Dark, concentric rings on leaves; yellowing of lower leaves.",
        "causes": "Fungus *Alternaria solani*.",
        "prevention": "Rotate crops and use resistant potato varieties.",
        "treatment": "Apply fungicides at the first sign of symptoms."
    },
    'Potato___Late_blight': {
        "name": "Late Blight in Potato",
        "symptoms": "Large, irregular greenish-brown lesions on leaves.",
        "causes": "Fungus *Phytophthora infestans*.",
        "prevention": "Use resistant varieties and avoid overhead irrigation.",
        "treatment": "Apply fungicides immediately upon detection."
    },
    'Potato___healthy': {
        "name": "Healthy Potato",
        "symptoms": "No visible disease; healthy foliage.",
        "causes": "N/A",
        "prevention": "Regular monitoring and proper care.",
        "treatment": "Continue standard potato management practices."
    },
    'Raspberry___healthy': {
        "name": "Healthy Raspberry",
        "symptoms": "No disease symptoms; vibrant green leaves.",
        "causes": "N/A",
        "prevention": "Ensure well-drained soil and proper sunlight.",
        "treatment": "Regular pruning and monitoring for pests."
    },
    'Soybean___healthy': {
        "name": "Healthy Soybean",
        "symptoms": "No visible disease; robust growth.",
        "causes": "N/A",
        "prevention": "Regular monitoring and balanced fertilization.",
        "treatment": "Maintain proper field sanitation."
    },
    'Squash___Powdery_mildew': {
        "name": "Powdery Mildew on Squash",
        "symptoms": "White, powdery spots on leaves and fruit.",
        "causes": "Fungal infection by *Podosphaera clandestina*.",
        "prevention": "Use resistant varieties and avoid overhead irrigation.",
        "treatment": "Apply sulfur-based fungicides during early spring."
    },
    'Strawberry___Leaf_scorch': {
        "name": "Leaf Scorch in Strawberry",
        "symptoms": "Brown, scorched edges on leaves.",
        "causes": "Environmental stress or fungal infection.",
        "prevention": "Ensure proper watering and avoid overcrowding.",
        "treatment": "Remove affected leaves and ensure proper care."
    },
    'Strawberry___healthy': {
        "name": "Healthy Strawberry",
        "symptoms": "No disease symptoms; healthy leaves and fruit.",
        "causes": "N/A",
        "prevention": "Regular monitoring and proper care.",
        "treatment": "Continue standard care practices."
    },
    'Tomato___Bacterial_spot': {
        "name": "Bacterial Spot on Tomato",
        "symptoms": "Small, dark lesions on leaves and fruit.",
        "causes": "Bacterial infection by *Xanthomonas campestris*.",
        "prevention": "Use resistant varieties and avoid overhead irrigation.",
        "treatment": "Apply copper-based bactericides during early spring."
    },
    'Tomato___Early_blight': {
        "name": "Early Blight on Tomato",
        "symptoms": "Dark concentric rings on older leaves.",
        "causes": "Fungus *Alternaria solani*.",
        "prevention": "Use resistant varieties and crop rotation.",
        "treatment": "Apply fungicides like chlorothalonil or copper-based sprays."
    },
    'Tomato___Late_blight': {
        "name": "Late Blight on Tomato",
        "symptoms": "Large, irregular, water-soaked spots on leaves.",
        "causes": "Fungus *Phytophthora infestans*.",
        "prevention": "Avoid overhead watering and remove infected plants.",
        "treatment": "Use appropriate fungicides early in infection."
    },
    'Tomato___Leaf_Mold': {
        "name": "Leaf Mold on Tomato",
        "symptoms": "Yellow patches on upper leaf surfaces with olive-green mold below.",
        "causes": "Fungus *Fulvia fulva*.",
        "prevention": "Provide good air circulation and reduce humidity.",
        "treatment": "Apply sulfur-based or copper-based fungicides."
    },
    'Tomato___Septoria_leaf_spot': {
        "name": "Septoria Leaf Spot",
        "symptoms": "Small, water-soaked spots on lower leaves.",
        "causes": "Fungus *Septoria lycopersici*.",
        "prevention": "Avoid overhead watering; use disease-free seeds.",
        "treatment": "Use fungicides like chlorothalonil or mancozeb."
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        "name": "Spider Mites on Tomato",
        "symptoms": "Yellow stippling and webbing on leaves.",
        "causes": "Infestation by *Tetranychus urticae*.",
        "prevention": "Encourage natural predators like ladybugs.",
        "treatment": "Use insecticidal soaps or neem oil."
    },
    'Tomato___Target_Spot': {
        "name": "Target Spot",
        "symptoms": "Circular lesions with concentric rings.",
        "causes": "Fungus *Corynespora cassiicola*.",
        "prevention": "Remove plant debris and rotate crops.",
        "treatment": "Use fungicides early in infection."
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        "name": "Tomato Yellow Leaf Curl Virus",
        "symptoms": "Upward curling of leaves, yellowing, stunted growth.",
        "causes": "Virus spread by whiteflies.",
        "prevention": "Control whitefly population and use resistant varieties.",
        "treatment": "No cure; remove infected plants immediately."
    },
    'Tomato___Tomato_mosaic_virus': {
        "name": "Tomato Mosaic Virus",
        "symptoms": "Mottled light/dark green on leaves, stunted growth.",
        "causes": "Virus *Tobamovirus*.",
        "prevention": "Use virus-free seeds and sterilize tools.",
        "treatment": "No cure; remove infected plants."
    },
    'Tomato___healthy': {
        "name": "Healthy Tomato",
        "symptoms": "No signs of disease; vibrant green leaves and fruit.",
        "causes": "N/A",
        "prevention": "Practice crop rotation and regular inspection.",
        "treatment": "Standard tomato care and monitoring."
    }
}
# Disease cures dictionary


# Paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "trained_model", "class_indices.json")

# Load model and class names
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(class_indices_path))

# Image preprocessing
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Predict function
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = class_indices[str(predicted_index)]
    return predicted_label

# Streamlit UI
st.title("🌿 Plant Disease Classifier")
st.markdown("Upload a plant leaf image and classify its disease.")

uploaded_image = st.file_uploader("Upload a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image.resize((224, 224)), caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button("🔍 Classify"):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f"🧠 Prediction: **{prediction}**")

            if prediction in disease_info:
                info = disease_info[prediction]
                st.markdown("### 🧾 Disease Information")
                st.markdown(f"**🪴 Name**: {info.get('name', 'N/A')}")
                st.markdown(f"**🔬 Symptoms**: {info.get('symptoms', 'N/A')}")
                st.markdown(f"**💊 Cure**: {info.get('treatment', 'N/A')}")
            else:
                st.warning("No additional information available for this disease.")
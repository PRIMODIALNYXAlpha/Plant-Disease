 🌱 Plant Disease Classifier

A Streamlit-based web app for detecting plant diseases using a trained TensorFlow model with 38 unique classes. Upload a leaf image and get instant diagnosis with disease info, treatment tips, and plant health advice.

🚀 Features

🌿 Classifies plant diseases from leaf images

📊 38 categories covering fruits, vegetables, and cereals

🧠 Powered by a TensorFlow CNN model

🧪 Interactive and responsive UI built with Streamlit

💡 Provides detailed metadata and treatment suggestions

🖼️ Supports image upload and preview

📈 Displays prediction confidence with bar chart

🧰 Tech Stack

Frontend: Streamlit

Backend: Python, TensorFlow

Model: Custom CNN trained on PlantVillage Dataset

Deployment: Streamlit Sharing / GitHub Pages (for static content)

🏷️ Classes

This model supports the following 38 plant-disease categories:

{0: 'Apple___Apple_scab',
 1: 'Apple___Black_rot',
 2: 'Apple___Cedar_apple_rust',
 3: 'Apple___healthy',
 4: 'Blueberry___healthy',
 5: 'Cherry_(including_sour)___Powdery_mildew',
 6: 'Cherry_(including_sour)___healthy',
 7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 8: 'Corn_(maize)___Common_rust_',
 9: 'Corn_(maize)___Northern_Leaf_Blight',
 10: 'Corn_(maize)___healthy',
 11: 'Grape___Black_rot',
 12: 'Grape___Esca_(Black_Measles)',
 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 14: 'Grape___healthy',
 15: 'Orange___Haunglongbing_(Citrus_greening)',
 16: 'Peach___Bacterial_spot',
 17: 'Peach___healthy',
 18: 'Pepper,_bell___Bacterial_spot',
 19: 'Pepper,_bell___healthy',
 20: 'Potato___Early_blight',
 21: 'Potato___Late_blight',
 22: 'Potato___healthy',
 23: 'Raspberry___healthy',
 24: 'Soybean___healthy',
 25: 'Squash___Powdery_mildew',
 26: 'Strawberry___Leaf_scorch',
 27: 'Strawberry___healthy',
 28: 'Tomato___Bacterial_spot',
 29: 'Tomato___Early_blight',
 30: 'Tomato___Late_blight',
 31: 'Tomato___Leaf_Mold',
 32: 'Tomato___Septoria_leaf_spot',
 33: 'Tomato___Spider_mites Two-spotted_spider_mite',
 34: 'Tomato___Target_Spot',
 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 36: 'Tomato___Tomato_mosaic_virus',
 37: 'Tomato___healthy'}

📦 Installation & Setup

# Clone the repository
git clone https://github.com/PRIMODIALNYXAlpha/Plant-Disease.git
cd Plant-Disease

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

🧪 Run Locally

streamlit run app.py

Then open http://localhost:8501 in your browser.

📂 Project Structure

Plant-Disease/
├── app.py                     # Main Streamlit app
├── model/
│   └── plant_disease_model.h5  # Trained TensorFlow model
├── utils/
│   └── disease_metadata.json   # Detailed info about each disease
├── examples/
│   └── sample_leaf.jpg        # Example input image
├── README.md
└── requirements.txt

📸 Screenshots

Add screenshots here of the web UI with prediction results.

🌐 Deployment

You can deploy using:

Streamlit Cloud – Free hosting for Streamlit apps

Render / Heroku – For more custom backend setups

Make sure to include your model file (plant_disease_model.h5) and metadata.

📌 TODO



🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

📄 License

MIT

👨‍💻 Author

Made with ❤️ by PRIMODIALNYXAlpha


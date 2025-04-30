from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

# Charger ton modèle CNN pré-entraîné
model = load_model("cnn_panneau_model.h5")

# Définir la taille d'entrée attendue par le modèle
IMG_SIZE = (150, 150)


# Fonction pour traiter l'image et faire la prédiction
def prepare_image(img: BytesIO):
    img = Image.open(img)
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normaliser l'image
    return img_array

# ➡ Route d'accueil
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de classification des panneaux solaires"}

# ➡ Route pour faire une prédiction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Lire l'image envoyée
        img_bytes = await file.read()
        img = BytesIO(img_bytes)

        # Préparer l'image et faire la prédiction
        img_array = prepare_image(img)
        
        # 🛠 Ajout : afficher la forme du tableau pour vérifier
        print("Forme de l'image envoyée :", img_array.shape)

        prediction = model.predict(img_array)
        
        # 🛠 Ajout : afficher la valeur brute de prédiction
        print("Valeur brute prédite :", prediction)

        # Supposons que 0 correspond à "Propre" et 1 à "Sale"
        result = "Sale" if prediction[0][0] > 0.5 else "Propre"

        return {"prediction": result}
    
    except Exception as e:
        # En cas d'erreur, afficher le problème
        print(f"Erreur : {e}")
        return {"error": str(e)}
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import json
from pathlib import Path
import lightgbm as lgb
from utils.patiente import extract_features_from_images

app = Flask(__name__)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "lightgbm_model.txt"
FEATURE_LIST_PATH = BASE_DIR / "model" / "selected_features.json"

# -----------------------------
# Load model & selected features
# -----------------------------
gbm = lgb.Booster(model_file=str(MODEL_PATH))

with open(FEATURE_LIST_PATH) as f:
    final_features = json.load(f)

# === champs que l'utilisateur doit remplir ===
clin_features_user = [
    'multifocal_cancer', 'hr', 'pr', 'her2', 'mammaprint', 'age', 'weight'
]

# === champs fixes définis dans le code ===
clin_features_fixed = [
    'num_phases', 'image_rows', 'num_slices', 'slice_thickness',
    'high_bit', 'window_center', 'window_width', 'echo_time', 'repetition_time'
]

radiomics_features = [f for f in final_features if f not in clin_features_user + clin_features_fixed]


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("pcr.html", clin_features=clin_features_user)

@app.route("/pcr/predict", methods=["POST"])
def predict():
    # Récupérer les inputs cliniques uniquement
    clinical_data = {f: float(request.form.get(f, 0)) for f in [
        'multifocal_cancer', 'hr', 'pr', 'her2', 'mammaprint', 'age', 'weight'
    ]}
    df_clin = pd.DataFrame([clinical_data])

    # Ajouter les features techniques fixes
    fixed_features = {
        'num_phases': 3,
        'image_rows': 448,
        'num_slices': 160,
        'slice_thickness': 1.5,
        'high_bit': 11,
        'window_center': 53,
        'window_width': 145,
        'echo_time': 4.12,
        'repetition_time': 1.36
    }
    df_fixed = pd.DataFrame([fixed_features])

    # Fusion features cliniques + fixes
    X_clinical = pd.concat([df_clin, df_fixed], axis=1)

    # Récupérer les dossiers d'images et segmentations
    images = request.files.getlist("images")
    segmentations = request.files.getlist("segmentations")

    if not images or not segmentations:
        return "Veuillez uploader les dossiers d'images et de segmentations."

    # Extraire features radiomics
    df_images = extract_features_from_images(images, segmentations)

    # Sélectionner seulement les features utilisées par LightGBM
    df_images = df_images.reindex(columns=radiomics_features, fill_value=0)

    # Fusion finale
    X_input = pd.concat([X_clinical, df_images[radiomics_features]], axis=1)

    # Prédiction
    y_prob = gbm.predict(X_input)
    y_pred = (y_prob >= 0.47).astype(int)  # seuil optimisé

    if y_pred[0] == 1:
        result_text = f"La patiente présente une probabilité élevée de réponse complète au traitement néoadjuvant (pCR = {y_prob[0]:.2f})."
    else:
        result_text = f"La patiente présente une probabilité faible de réponse complète au traitement néoadjuvant (pCR = {y_prob[0]:.2f})."

    return result_text


if __name__ == "__main__":
    app.run(debug=True, port=5001)
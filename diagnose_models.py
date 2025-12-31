import json
import joblib
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model

print('Chargement des fichiers depuis models/...')
scaler = joblib.load('models/scaler.pkl')
mlp = load_model('models/mlp_model.h5')
with open('models/feature_names.json') as f:
    features = json.load(f)

print('Scaler n_features_in_:', getattr(scaler, 'n_features_in_', None))
print('Model input shape:', mlp.input_shape)
print('Feature names count:', len(features))
print('Feature names sample:', features[:5])

# Charger quelques lignes malignes du dataset si disponible
if os.path.exists('data/data.csv'):
    df = pd.read_csv('data/data.csv')
    if 'diagnosis' in df.columns:
        df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
        X = df.drop(columns=['diagnosis'])
        y = df['diagnosis']
        malignant = X[y==1]
        if len(malignant) > 0:
            sample = malignant.iloc[0].values.reshape(1, -1)
            # Assurer l'ordre si features names différent
            try:
                X_ord = X[features]
                sample = X_ord[y==1].iloc[0].values.reshape(1, -1)
            except Exception:
                pass

            sample_scaled = scaler.transform(sample)
            proba = mlp.predict(sample_scaled, verbose=0)
            print('Exemple malignant - probabilité (malignant):', float(proba[0][0]))
        else:
            print('Pas d\'échantillon malignant trouvé dans data/data.csv')
    else:
        print('Colonne diagnosis absente dans data/data.csv')
else:
    print('data/data.csv introuvable')

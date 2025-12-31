# feature_selection.py
# =====================================================
# FEATURE EXTRACTION + SELECTION ROBUSTE (MAMA-MIA 2025)
# Objectif : prédire pCR au NACT
# =====================================================

import pandas as pd
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from radiomics import featureextractor
import warnings
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ====================== CHEMINS ======================
IMG_DIR = Path(r"C:\Users\user1\MAMA-MIA\images")
SEG_DIR = Path(r"C:\Users\user1\MAMA-MIA2\segmentations\expert")

CSV_IDS = Path(r"C:\EYA\4eme\ML\projetML\selected_1071_cases.csv")
CLINICAL_FILE = Path(r"C:\EYA\4eme\ML\projetML\clinical_and_imaging_info.xlsx")

OUT_DIR = Path(r"C:\EYA\4eme\ML\projetML\features_selection1")
OUT_DIR.mkdir(exist_ok=True)

# ====================== RADIOMICS EXTRACTOR ======================
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.settings["binWidth"] = 25
extractor.settings["resampledPixelSpacing"] = [1, 1, 1]
extractor.settings["interpolator"] = "sitkBSpline"
extractor.enableAllFeatures()

print("Démarrage extraction radiomiques...")

data = []

# ====================== EXTRACTION ======================
for folder in IMG_DIR.iterdir():
    if not folder.is_dir():
        continue

    pid = folder.name

    all_files = [f for f in folder.rglob("*") if f.is_file()]
    if not all_files:
        continue

    priority_keywords = [
        "post", "phase1", "phase_1", "t1post",
        "postcontrast", "adc", "sub", "dyn", "contrast"
    ]

    phase1 = None
    for f in all_files:
        if any(kw in f.name.lower() for kw in priority_keywords):
            phase1 = f
            break

    if phase1 is None:
        phase1 = max(all_files, key=lambda x: x.stat().st_size)

    mask_candidates = list(SEG_DIR.rglob(f"*{pid}*"))
    if not mask_candidates:
        continue

    mask = mask_candidates[0]

    try:
        img = sitk.ReadImage(str(phase1))
        msk = sitk.ReadImage(str(mask))

        features = extractor.execute(img, msk)
        row = {"patient_id": pid}

        for k, v in features.items():
            if k.startswith("diagnostics"):
                continue
            if isinstance(v, (int, float, np.integer, np.floating)):
                row[k] = float(v)
            else:
                row[k] = np.nan

        data.append(row)

    except Exception as e:
        print(f"Erreur extraction {pid} : {e}")

print(f"\nExtraction terminée : {len(data)} patients")

# ====================== DATAFRAME ======================
df = pd.DataFrame(data).set_index("patient_id")

df = df.apply(pd.to_numeric, errors="coerce")
print(f"Patients avec features extraites : {df.shape[0]}")

# ====================== ALIGNEMENT CSV ======================
csv_ids = pd.read_csv(CSV_IDS)["patient_id"].astype(str)
df = df.loc[df.index.intersection(csv_ids)]

print(f"Patients communs CSV ∩ radiomics : {df.shape[0]}")

# ====================== LOAD TRUE pCR LABELS ======================
df_clin = pd.read_excel(CLINICAL_FILE, sheet_name="dataset_info")
df_clin["patient_id"] = df_clin["patient_id"].astype(str)

y = (
    df_clin
    .set_index("patient_id")
    .loc[df.index, "pcr"]
    .dropna()
    .astype(int)
)

# Align X with y
df = df.loc[y.index]

print(f"Patients avec labels pCR : {df.shape[0]}")
print("Distribution pCR :")
print(y.value_counts())

# ====================== NETTOYAGE ======================
print("Nettoyage des features...")

df = df.dropna(axis=1, how="all")
df = df.loc[:, df.nunique() > 1]
df = df.fillna(df.median())

scaler = StandardScaler()
X = pd.DataFrame(
    scaler.fit_transform(df),
    index=df.index,
    columns=df.columns
)

print(f"Dataset final propre : {X.shape}")

# ====================== FEATURE SELECTION ======================
print("Feature selection supervisée en cours...")

# 1️⃣ Variance Threshold
sel1 = VarianceThreshold(threshold=0.01)
X1 = sel1.fit_transform(X)
cols1 = X.columns[sel1.get_support()]

# 2️⃣ ANOVA (top 300)
k_anova = min(300, X1.shape[1])
sel2 = SelectKBest(f_classif, k=k_anova)
X2 = sel2.fit_transform(X1, y)
cols2 = cols1[sel2.get_support()]

# 3️⃣ Random Forest importance (top 100)
rf = RandomForestClassifier(
    n_estimators=1000,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf.fit(X2, y)

importances = rf.feature_importances_
top_idx = np.argsort(importances)[-100:]
final_features = np.array(cols2)[top_idx]

df_final = X[final_features]

# ====================== SAVE ======================
df.to_csv(OUT_DIR / "features_all.csv")
df_final.to_csv(OUT_DIR / "features_selected_100.csv")

print("\n100 features sélectionnées ✔")

# ====================== TOP 20 PLOT ======================
top20_names = final_features[-20:][::-1]
top20_importances = importances[top_idx][-20:][::-1]

plt.figure(figsize=(10, 8))
plt.barh(range(20), top20_importances)
plt.yticks(range(20), top20_names, fontsize=9)
plt.xlabel("Importance (Random Forest)")
plt.title("Top 20 radiomics features – pCR prediction")
plt.tight_layout()
plt.savefig(OUT_DIR / "top20_features.png", dpi=200)
plt.show()

print("\nTout est prêt ✔")
print("→ features_all.csv")
print("→ features_selected_100.csv")
print("→ top20_features.png")

os.startfile(str(OUT_DIR))
input("\nAppuie sur Entrée pour fermer...")

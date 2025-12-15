# utils/feature_extraction.py
import pandas as pd
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from radiomics import featureextractor
from sklearn.preprocessing import StandardScaler
from tempfile import TemporaryDirectory

def extract_features_from_images(images, segmentations):
    """
    images, segmentations : listes de fichiers uploadés depuis Flask
    Retourne : DataFrame avec les features radiomics, index = patient_id
    """
    # ====================== Radiomics extractor ======================
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings["binWidth"] = 25
    extractor.settings["resampledPixelSpacing"] = [1,1,1]
    extractor.settings["interpolator"] = "sitkBSpline"
    extractor.enableAllFeatures()

    data = []

    # Trier les fichiers pour correspondre
    images = sorted(images, key=lambda f: f.filename)
    segmentations = sorted(segmentations, key=lambda f: f.filename) if segmentations else [None]*len(images)

    # Utilisation d'un dossier temporaire pour enregistrer les uploads
    with TemporaryDirectory() as tmpdir:
        for idx, (img_file, seg_file) in enumerate(zip(images, segmentations)):
            pid = f"patient_{idx+1}"
            try:
                # Sauvegarder l'image
                img_path = Path(tmpdir) / img_file.filename
                img_file.save(img_path)

                # Sauvegarder la segmentation si fournie
                seg_path = None
                if seg_file:
                    seg_path = Path(tmpdir) / seg_file.filename
                    seg_file.save(seg_path)

                # Lire avec SimpleITK
                img = sitk.ReadImage(str(img_path))
                seg = sitk.ReadImage(str(seg_path)) if seg_path else None

                features = extractor.execute(img, seg) if seg else extractor.execute(img)
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
                print(f"Erreur extraction pour {pid}: {e}")

    # ====================== DataFrame ======================
    if not data:
        print("Aucune feature extraite !")
        return pd.DataFrame()

    df = pd.DataFrame(data).set_index("patient_id")
    df = df.apply(pd.to_numeric, errors="coerce")

    # Nettoyage
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, df.nunique() > 1]
    df = df.fillna(df.median())

    # Standardisation
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )

    return df_scaled

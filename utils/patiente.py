import pandas as pd
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from radiomics import featureextractor
from sklearn.preprocessing import StandardScaler
from tempfile import TemporaryDirectory


def extract_features_from_images(images, segmentations):
    """
    images, segmentations :
    - Flask FileStorage
    - FastAPI UploadFile sauvegardés
    - pathlib.Path

    Retourne : DataFrame standardisé des features radiomics
    """

    # ====================== Radiomics extractor ======================
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings["binWidth"] = 25
    extractor.settings["resampledPixelSpacing"] = [1, 1, 1]
    extractor.settings["interpolator"] = "sitkBSpline"
    extractor.enableAllFeatures()

    data = []

    # ====================== Alignement fichiers ======================
    def get_name(f):
        return f.filename if hasattr(f, "filename") else Path(f).name

    images = sorted(images, key=get_name)

    if segmentations:
        segmentations = sorted(segmentations, key=get_name)
    else:
        segmentations = [None] * len(images)

    # ====================== Temp directory ======================
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for idx, (img, seg) in enumerate(zip(images, segmentations)):
            pid = f"patient_{idx+1}"

            try:
                # ---------------- IMAGE ----------------
                if hasattr(img, "save"):  # Flask FileStorage
                    img_path = tmpdir / img.filename
                    img.save(img_path)
                else:  # Path
                    img_path = Path(img)

                # ---------------- SEGMENTATION ----------------
                seg_path = None
                if seg:
                    if hasattr(seg, "save"):
                        seg_path = tmpdir / seg.filename
                        seg.save(seg_path)
                    else:
                        seg_path = Path(seg)

                # ---------------- READ ----------------
                img_itk = sitk.ReadImage(str(img_path))
                seg_itk = sitk.ReadImage(str(seg_path)) if seg_path else None

                features = (
                    extractor.execute(img_itk, seg_itk)
                    if seg_itk else
                    extractor.execute(img_itk)
                )

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
                print(f"❌ Radiomics error for {pid}: {e}")

    # ====================== DataFrame ======================
    if not data:
        print("❌ No radiomics features extracted")
        return pd.DataFrame()

    df = pd.DataFrame(data).set_index("patient_id")
    df = df.apply(pd.to_numeric, errors="coerce")

    # ---------------- CLEANING ----------------
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, df.nunique() > 1]

    # 🔥 FIX CRITIQUE
    if df.shape[1] == 0:
        print("❌ Radiomics DF has zero valid features")
        return pd.DataFrame()

    df = df.fillna(df.median())

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )


    df = df.fillna(df.median())

    # ---------------- SCALING ----------------
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )

    return df_scaled

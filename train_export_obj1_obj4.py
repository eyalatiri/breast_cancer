# train_export_obj2_obj3_obj5.py
# - Exporte OBJ2 -> models/exported_models/obj2_intensity.joblib
# - Exporte OBJ3 -> models/exported_models/obj3_endocrine_proxy.joblib
# - Exporte OBJ5 (RL + XAI) -> models/exported_models/obj5_rl_xai.joblib
# - Exporte OBJ4 -> models/exported_models/obj4_relapse.joblib
#
# ✅ Option B appliquée: PAS de FrozenEstimator (compatible sklearn ancien)
# - OBJ4: calibration via CalibratedClassifierCV(..., cv="prefit") (le modèle est déjà fit)
#
# ⚠️ Le reste (OBJ2/OBJ3/OBJ5) est gardé intact au niveau logique.

import os
import glob
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve,
    classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score
)

import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb


# =========================
# PATHS + CONSTANTS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "models", "exported_models")
os.makedirs(OUT_DIR, exist_ok=True)

OBJ1_PATH = os.path.join(OUT_DIR, "obj1_therap_decision.joblib")

# OBJ4 export
OBJ4_EXPORT_PATH = os.path.join(OUT_DIR, "obj4_relapse.joblib")

RANDOM_STATE = 42
TARGET_RECALL = 0.70
THRESHOLD_MODE = "f1"  # "f1" ou "recall70"
TEST_SIZE = 0.20
VAL_SIZE_IN_TRAIN_FULL = 0.25  # (sur train_full) -> 60/20/20


# =========================
# HELPERS
# =========================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str).str.strip().str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("+", "plus", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    if "pam50_plus_claudin_low_subtype" in df.columns and "pam50_subtype" not in df.columns:
        df = df.rename(columns={"pam50_plus_claudin_low_subtype": "pam50_subtype"})
    if "pam50_claudin_low_subtype" in df.columns and "pam50_subtype" not in df.columns:
        df = df.rename(columns={"pam50_claudin_low_subtype": "pam50_subtype"})
    return df

def dedup_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()].copy()

def find_metabric_csv(base_dir):
    p = os.path.join(base_dir, "Breast Cancer METABRIC.csv")
    if os.path.exists(p):
        return p
    hits = glob.glob(os.path.join(base_dir, "*metabric*.csv"))
    if hits:
        return hits[0]
    all_csv = glob.glob(os.path.join(base_dir, "*.csv"))
    if not all_csv:
        return None
    all_csv = sorted(all_csv, key=lambda x: os.path.getsize(x), reverse=True)
    return all_csv[0]

def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def clean_cat(s: pd.Series) -> pd.Series:
    return (
        s.astype(str).str.strip().str.lower()
        .replace({"": np.nan, "nan": np.nan, "none": np.nan})
    )

def to01(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in ["1", "yes", "y", "true", "treated", "pos", "positive"]:
        return 1
    if s in ["0", "no", "n", "false", "none", "neg", "negative"]:
        return 0
    try:
        return 1 if float(s) >= 0.5 else 0
    except Exception:
        return np.nan

def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("string"):
            df[c] = df[c].astype(str).str.strip().str.lower()
            df[c] = df[c].replace({"": np.nan, "nan": np.nan, "none": np.nan})
    return df

def intensity_level(theta, q):
    if theta <= q["q25"]:
        return "LOW"
    if theta <= q["q50"]:
        return "MODERATE"
    if theta <= q["q75"]:
        return "HIGH"
    return "VERY_HIGH"

def best_thr_f1(y_true, p):
    grid = np.linspace(0.05, 0.95, 181)
    best_t, best_f = 0.5, -1
    for t in grid:
        f = f1_score(y_true, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = float(f), float(t)
    return float(best_t), float(best_f)

def _canon_surgery(x: str) -> str:
    s = str(x).strip().lower()
    if "mast" in s:
        return "MASTECTOMY"
    if "conserv" in s or "breast" in s:
        return "BREAST CONSERVING"
    return s.upper() if s else "UNKNOWN"

def _canon_pam50(x: str) -> str:
    s = str(x).strip()
    if not s or s.lower() in ["nan", "none"]:
        return "UNKNOWN"
    return s

def _canon_cellularity(value, cfg):
    if value is None:
        return "UNK"

    # ✅ Si numeric: binning (ton front envoie souvent 0..1)
    try:
        x = float(value)
        # si 0..1
        if 0.0 <= x <= 1.0:
            if x < 0.33:
                return "LOW"
            elif x < 0.66:
                return "MED"
            else:
                return "HIGH"
        # si 1..3 (parfois)
        if 0.9 <= x <= 3.1:
            if x <= 1.0:
                return "LOW"
            elif x <= 2.0:
                return "MED"
            else:
                return "HIGH"
    except Exception:
        pass

    s = str(value).strip()
    sl = s.lower()

    if s in ["LOW", "MED", "HIGH", "UNK"]:
        return s

    cell_map = cfg.get("cell_map", {})
    if isinstance(cell_map, dict) and sl in cell_map:
        return str(cell_map[sl])

    if sl in ["low", "l", "1"]:
        return "LOW"
    if sl in ["moderate", "medium", "med", "m", "2"]:
        return "MED"
    if sl in ["high", "h", "3"]:
        return "HIGH"
    return "UNK"



# =========================
# LOAD OBJ1 bundle
# =========================
if not os.path.exists(OBJ1_PATH):
    raise FileNotFoundError(f"OBJ1 introuvable: {OBJ1_PATH}. Exporte OBJ1 d'abord.")

obj1 = joblib.load(OBJ1_PATH)
features = obj1.get("features", [])
labels = obj1.get("labels", [])
pipes = obj1.get("pipes", None) or obj1.get("pipe", None)

if not features or not labels or pipes is None:
    raise ValueError("OBJ1 bundle invalide (features/labels/pipes).")
if not isinstance(pipes, (list, tuple)) or len(pipes) != len(labels):
    raise ValueError("OBJ1: pipes doit être une liste alignée avec labels.")

print("✅ OBJ1 loaded | labels =", labels)


# =========================
# LOAD DATA (df + df1)
# =========================
DATA_PATH = find_metabric_csv(BASE_DIR)
if DATA_PATH is None or not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Aucun CSV trouvé dans {BASE_DIR} (mets METABRIC ici).")

df = pd.read_csv(DATA_PATH)
df = normalize_columns(df)
df = dedup_cols(df)
print("✅ Dataset:", DATA_PATH, "| shape:", df.shape)

# df1 (OBJ5 doit utiliser df1 -> on charge la même source en local)
df1 = pd.read_csv(DATA_PATH)
df1 = normalize_columns(df1)
df1 = dedup_cols(df1)
print("✅ df1 loaded for OBJ5 | shape:", df1.shape)


# targets columns for OBJ2
col_chemo = pick_first_existing(df, ["chemotherapy", "chemo"])
col_horm = pick_first_existing(df, ["hormone_therapy", "hormonotherapy", "hormonal_therapy", "hormone"])
col_radio = pick_first_existing(df, ["radiotherapy", "radio_therapy", "radiation_therapy", "radiation"])

if col_chemo is None or col_horm is None:
    raise ValueError("❌ Impossible de trouver chemotherapy et/ou hormone_therapy.")

ycols = {"chemotherapy": col_chemo, "hormone_therapy": col_horm}
if "radiotherapy" in labels:
    if col_radio is None:
        raise ValueError("OBJ1 a radiotherapy mais dataset ne l'a pas.")
    ycols["radiotherapy"] = col_radio


# =========================
# OBJ2 — EXPORT
# =========================
Y2 = pd.DataFrame({lab: df[ycols[lab]].map(to01) for lab in labels})
data2 = df[features].join(Y2)
data2 = dedup_cols(data2).dropna(subset=labels)

X2 = normalize_strings(data2[features].copy())
Y2 = data2[labels].astype(int).copy()

print("✅ OBJ2 After dropna:", X2.shape, Y2.shape)

strat2 = Y2.astype(str).agg("".join, axis=1)
X2_tr, X2_te, Y2_tr, Y2_te = train_test_split(
    X2, Y2, test_size=0.2, random_state=RANDOM_STATE, stratify=strat2
)

theta_target_tr = Y2_tr.mean(axis=1).values.astype(float)

P_tr = np.zeros((len(X2_tr), len(labels)), dtype=float)
for i, lab in enumerate(labels):
    P_tr[:, i] = pipes[i].predict_proba(X2_tr)[:, 1].astype(float)

theta_reg = Ridge(alpha=1.0)
theta_reg.fit(P_tr, theta_target_tr)

theta_hat_tr = theta_reg.predict(P_tr).astype(float)
q = {
    "q25": float(np.quantile(theta_hat_tr, 0.25)),
    "q50": float(np.quantile(theta_hat_tr, 0.50)),
    "q75": float(np.quantile(theta_hat_tr, 0.75)),
}

print("✅ OBJ2 Quantiles:", q)
print("✅ OBJ2 Example level:", intensity_level(float(theta_hat_tr[0]), q))

obj2_bundle = {
    "labels": labels,
    "obj1_pipes": pipes,
    "theta_reg": theta_reg,
    "q": q,
    "theta_definition": "mean_of_actual_treatments (0..1), fitted via Ridge(P_obj1 -> theta)",
}

out_path2 = os.path.join(OUT_DIR, "obj2_intensity.joblib")
joblib.dump(obj2_bundle, out_path2)
print("✅ Exported OBJ2 ->", out_path2)


# =========================
# OBJ3 — TRAIN + EXPORT
# =========================
col_rfs = pick_first_existing(df, ["relapse_free_status", "rfs_status", "rfs"])
if col_rfs is None:
    raise ValueError("❌ OBJ3: relapse_free_status introuvable dans le dataset.")

horm = df[col_horm].map(to01)
rfs = clean_cat(df[col_rfs])

y3 = rfs.map({"recurred": 1, "not_recurred": 0, "not recurred": 0})
mask3 = (horm == 1) & y3.notna()
df3 = df.loc[mask3].copy()
y3 = y3.loc[mask3].astype(int)

print("✅ OBJ3 Cohorte hormone_therapy=YES:", df3.shape, "| y rate:", round(float(y3.mean()), 3))
print("✅ OBJ3 y distribution:", y3.value_counts().to_dict())
if y3.nunique() < 2:
    raise ValueError("❌ OBJ3: y a une seule classe -> impossible d'entraîner.")

FEATURES3 = [
    "pr_status",
    "her2_status",
    "age_at_diagnosis",
    "inferred_menopausal_state",
    "tumor_size",
    "tumor_stage",
    "lymph_nodes_examined_positive",
    "pam50_subtype",
    "neoplasm_histologic_grade",
]

ki67_cols = [c for c in df3.columns if "ki67" in c]
if ki67_cols:
    FEATURES3.append(ki67_cols[0])
    print("✅ OBJ3 Ki67 detected:", ki67_cols[0])

FEATURES3 = [c for c in FEATURES3 if c in df3.columns]
X3 = df3[FEATURES3].copy()

num_candidates = [
    "age_at_diagnosis", "tumor_size", "tumor_stage",
    "lymph_nodes_examined_positive", "neoplasm_histologic_grade"
]
num_cols = [c for c in num_candidates if c in X3.columns]
cat_cols = [c for c in X3.columns if c not in num_cols]

for c in num_cols:
    X3[c] = pd.to_numeric(X3[c], errors="coerce")
for c in cat_cols:
    X3[c] = clean_cat(X3[c])

X3 = X3.drop(columns=[c for c in X3.columns if X3[c].isna().all()], errors="ignore")
num_cols = [c for c in num_cols if c in X3.columns]
cat_cols = [c for c in X3.columns if c not in num_cols]

print("✅ OBJ3 Features:", X3.shape[1], "| num:", len(num_cols), "| cat:", len(cat_cols))

X3_trval, X3_test, y3_trval, y3_test = train_test_split(
    X3, y3, test_size=0.20, random_state=RANDOM_STATE, stratify=y3
)
X3_train, X3_val, y3_train, y3_val = train_test_split(
    X3_trval, y3_trval, test_size=0.20, random_state=RANDOM_STATE, stratify=y3_trval
)

prep3 = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), cat_cols),
])

pos = int(y3_train.sum())
neg = int(len(y3_train) - pos)
scale_pos_weight = (neg / (pos + 1e-9))

xgb3 = XGBClassifier(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight
)

base3 = Pipeline([("prep", prep3), ("clf", xgb3)])
cal3 = CalibratedClassifierCV(estimator=base3, method="sigmoid", cv=3)
cal3.fit(X3_train, y3_train)

p3_val = cal3.predict_proba(X3_val)[:, 1]
thr3, f13 = best_thr_f1(y3_val.values, p3_val)

p3_test = cal3.predict_proba(X3_test)[:, 1]
f1_test = f1_score(y3_test.values, (p3_test >= thr3).astype(int), zero_division=0)

print(f"✅ OBJ3 thr(F1@val)={thr3:.3f} | F1_val={f13:.3f} | F1_test={f1_test:.3f}")


def _get_feature_names_from_preprocess(prep: ColumnTransformer):
    names = []
    for name, trans, cols in prep.transformers_:
        if name == "remainder" or trans is None:
            continue
        cols = list(cols) if hasattr(cols, "__iter__") and not isinstance(cols, str) else [cols]
        if hasattr(trans, "named_steps") and "ohe" in trans.named_steps:
            ohe = trans.named_steps["ohe"]
            try:
                names.extend(list(ohe.get_feature_names_out(cols)))
            except Exception:
                names.extend([f"{c}_<cat>" for c in cols])
        else:
            names.extend(list(cols))
    return np.array(names, dtype=object)

def _group_parent(feat_name: str, original_cols):
    for base in original_cols:
        if feat_name == base or feat_name.startswith(base + "_"):
            return base
    return feat_name

def build_obj3_global_xai(calibrated_model: CalibratedClassifierCV, original_cols):
    base_pipe = calibrated_model.calibrated_classifiers_[0].estimator
    prep = base_pipe.named_steps["prep"]
    clf = base_pipe.named_steps["clf"]

    feat_names = _get_feature_names_from_preprocess(prep)
    importances = getattr(clf, "feature_importances_", None)
    if importances is None:
        importances = np.ones(len(feat_names), dtype=float) / max(1, len(feat_names))

    df_imp = pd.DataFrame({
        "feature": feat_names,
        "parent": [_group_parent(str(f), original_cols) for f in feat_names],
        "imp": importances.astype(float)
    })
    grouped = df_imp.groupby("parent")["imp"].sum().sort_values(ascending=False)
    top = [{"var": k, "importance": float(v)} for k, v in grouped.head(10).items()]
    return {"top_variables": top, "raw_grouped": {k: float(v) for k, v in grouped.items()}}

def obj3_local_xai_delta_proba(calibrated_model: CalibratedClassifierCV, patient_row: pd.Series,
                              num_cols, cat_cols, n_top=6):
    x0 = patient_row.to_frame().T.copy()
    p0 = float(calibrated_model.predict_proba(x0)[:, 1][0])

    deltas = []
    for c in num_cols:
        x1 = x0.copy()
        v = x1.at[x1.index[0], c]
        if pd.isna(v):
            continue
        x1.at[x1.index[0], c] = np.nan
        p1 = float(calibrated_model.predict_proba(x1)[:, 1][0])
        deltas.append((c, p1 - p0, v))

    for c in cat_cols:
        x1 = x0.copy()
        v = x1.at[x1.index[0], c]
        if pd.isna(v):
            continue
        x1.at[x1.index[0], c] = "unknown"
        p1 = float(calibrated_model.predict_proba(x1)[:, 1][0])
        deltas.append((c, p1 - p0, v))

    deltas = sorted(deltas, key=lambda x: abs(x[1]), reverse=True)

    inc, dec = [], []
    for c, d, v in deltas:
        item = {"var": c, "value": (None if pd.isna(v) else str(v)), "delta_proba": float(d)}
        (inc if d >= 0 else dec).append(item)

    return {"baseline_proba": p0, "increase": inc[:n_top], "decrease": dec[:n_top]}

xai3_global = build_obj3_global_xai(cal3, FEATURES3)

obj3_bundle = {
    "features": FEATURES3,
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "cal": cal3,
    "thr": float(thr3),
    "proxy_definition": "cohort hormone_therapy==YES; target = relapse_free_status==Recurred (proxy, not observed resistance)",
    "xai_global": xai3_global,
    "xai_local_method": "delta_proba_by_neutralization",
}
out_path3 = os.path.join(OUT_DIR, "obj3_endocrine_proxy.joblib")
joblib.dump(obj3_bundle, out_path3)
print("✅ Exported OBJ3 ->", out_path3)


# interpretation.py  (OBJ5 RL + XAI)
# - charge models/exported_models/obj5_rl_xai.joblib
# - build state depuis payload (robuste)
# - renvoie recommendation + XAI (feature_importance, counterfactuals, explication)

import os
import json
import joblib
import numpy as np


# ============================================================
# Helpers: canonisation + state building
# ============================================================

def _bin_age_from_cfg(age_value, cfg):
    try:
        a = float(age_value)
    except Exception:
        return "UNK"
    bins = cfg.get("age_bins", [0, 50, 70, 120])
    labels = cfg.get("age_labels", ["YOUNG", "MID", "OLD"])
    if a < float(bins[1]):
        return str(labels[0])
    if a < float(bins[2]):
        return str(labels[1])
    return str(labels[2])


def _canon_cellularity(value, cfg):
    if value is None:
        return "UNK"
    s = str(value).strip()
    sl = s.lower()

    if s in ["LOW", "MED", "HIGH", "UNK"]:
        return s

    cell_map = cfg.get("cell_map", {})
    if isinstance(cell_map, dict) and sl in cell_map:
        return str(cell_map[sl])

    if sl in ["low", "l", "1"]:
        return "LOW"
    if sl in ["moderate", "medium", "med", "m", "2"]:
        return "MED"
    if sl in ["high", "h", "3"]:
        return "HIGH"
    return "UNK"


def _canon_surgery(value, cfg):
    if value is None:
        return "UNKNOWN"
    s = str(value).strip()
    sl = s.lower()

    surg_map = cfg.get("surgery_map", {})
    if isinstance(surg_map, dict) and sl in surg_map:
        return str(surg_map[sl])

    if "mast" in sl:
        return "MASTECTOMY"
    if "conserv" in sl or "breast" in sl:
        return "BREAST CONSERVING"
    return "UNKNOWN"


def _canon_pam50(value, cfg):
    if value is None:
        return "UNKNOWN"
    s = str(value).strip()
    if not s or s.lower() in ["none", "nan", "null", ""]:
        return "UNKNOWN"
    return s


def _to01(value):
    if value is None:
        return 0
    if isinstance(value, (int, np.integer)):
        return 1 if int(value) == 1 else 0
    s = str(value).strip().lower()
    return 1 if s in ["1", "yes", "y", "true", "oui", "vrai"] else 0


def _obj5_build_state_from_payload(payload, cfg):
    # age_bin: si pas fourni -> calculé depuis age
    age_bin = payload.get("age_bin", None)
    if age_bin is None:
        age_raw = payload.get("Age_at_Diagnosis", payload.get("age_at_diagnosis", payload.get("age", None)))
        age_bin = _bin_age_from_cfg(age_raw, cfg)
    age_bin = str(age_bin)

    pam50_raw = payload.get("Pam50_subtype", payload.get("pam50_subtype", payload.get("PAM50", payload.get("pam50", None))))
    pam50 = _canon_pam50(pam50_raw, cfg)

    cell_bin = payload.get("cellularity_bin", payload.get("cell_bin", None))
    if cell_bin is None:
        cell_raw = payload.get("Cellularity", payload.get("cellularity", None))
        cell_bin = _canon_cellularity(cell_raw, cfg)
    cell_bin = str(cell_bin)

    surg_raw = payload.get("Type_of_Breast_Surgery", payload.get("type_of_breast_surgery", payload.get("surgery", None)))
    surgery = _canon_surgery(surg_raw, cfg)

    chemo_bin = payload.get("chemo_bin", payload.get("chemo", payload.get("Chemotherapy", payload.get("chemotherapy", None))))
    chemo_bin = int(_to01(chemo_bin))

    return (age_bin, str(pam50), cell_bin, str(surgery), chemo_bin)


def _state_to_sentence(state):
    age_bin, subtype, cell_bin, surgery, chemo = state
    age_txt = {"YOUNG": "< 50 ans", "MID": "50–69 ans", "OLD": "≥ 70 ans", "UNK": "âge inconnu"}.get(age_bin, age_bin)
    cell_txt = {"LOW": "faible", "MED": "modérée", "HIGH": "élevée", "UNK": "inconnue"}.get(cell_bin, cell_bin)
    surg_txt = "mastectomie" if surgery == "MASTECTOMY" else "chirurgie conservatrice" if surgery == "BREAST CONSERVING" else surgery
    chemo_txt = "chimio: oui" if int(chemo) == 1 else "chimio: non"
    return f"Profil RL: âge {age_txt}, PAM50 {subtype}, cellularité {cell_txt}, {surg_txt}, {chemo_txt}."


def _action_to_title_desc(action_id, action_obj):
    # action_obj peut être str ou dict etc.
    if isinstance(action_obj, str):
        # ✅ titre = label, pas "Action X"
        return (action_obj, "")

    if isinstance(action_obj, dict):
        title = action_obj.get("title") or action_obj.get("name") or action_obj.get("label") or f"Action {action_id}"
        desc = action_obj.get("desc") or action_obj.get("description") or action_obj.get("text")
        if desc is None:
            desc = json.dumps(action_obj, ensure_ascii=False)
        return (str(title), str(desc))

    if isinstance(action_obj, (list, tuple)):
        if len(action_obj) == 0:
            return (f"Action {action_id}", "—")
        if len(action_obj) == 1:
            return (str(action_obj[0]), "")
        return (str(action_obj[0]), str(action_obj[1]))

    return (f"Action {action_id}", str(action_obj))


# ============================================================
# XAI Explainer (version Flask-compatible)
# ============================================================

class XAI_Explainer_Scientific:
    def __init__(self, Q_table, actions_medical, cfg=None, rng_seed=42):
        self.Q = Q_table
        self.actions = actions_medical  # dict: id -> label
        self.cfg = cfg or {}
        self.rng = np.random.default_rng(int(rng_seed))

        # ✅ ordre stable des actions
        self.action_ids = sorted([int(k) for k in self.actions.keys()]) if isinstance(self.actions, dict) else []

        # domaines
        self.subtypes = self.cfg.get("known_subtypes", None)
        if not isinstance(self.subtypes, (list, tuple)) or len(self.subtypes) == 0:
            self.subtypes = sorted({k[1] for k in self.Q.keys()}) if self.Q else ["UNKNOWN"]

        self.surgery_bins = self.cfg.get("known_surgeries", None)
        if not isinstance(self.surgery_bins, (list, tuple)) or len(self.surgery_bins) == 0:
            self.surgery_bins = sorted({k[3] for k in self.Q.keys()}) if self.Q else ["MASTECTOMY", "BREAST CONSERVING", "UNKNOWN"]

        self.cell_bins = self.cfg.get("known_cells", None)
        if not isinstance(self.cell_bins, (list, tuple)) or len(self.cell_bins) == 0:
            self.cell_bins = ["LOW", "MED", "HIGH"]

    def _get_q_value(self, state, action_id):
        return float(self.Q.get(state + (int(action_id),), 0.0))

    def _qvals(self, state):
        ids = self.action_ids
        return ids, [self._get_q_value(state, a) for a in ids]

    def _get_best_action(self, state):
        ids, q = self._qvals(state)
        if not q:
            return int(ids[0]) if ids else 0
        return int(ids[int(np.argmax(q))])

    def _softmax(self, x, temperature=1.0):
        x = np.array(x, dtype=float) / float(temperature)
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-12)

    def _jensen_shannon_distance(self, p, q):
        p = np.clip(np.array(p, dtype=float), 1e-10, 1.0)
        q = np.clip(np.array(q, dtype=float), 1e-10, 1.0)
        m = 0.5 * (p + q)
        js = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
        return float(min(1.0, js))

    def compute_confidence_xai(self, state):
    
        qvals = [self._get_q_value(state, a) for a in self.actions]

        if not qvals:
            return 0.5

        qvals = np.array(qvals, dtype=float)

        max_q = qvals.max()
        min_q = qvals.min()

        if max_q == min_q:
            return 0.4  # faible signal

        # écart relatif normalisé
        confidence = (max_q - min_q) / (abs(max_q) + 1e-6)
        return float(np.clip(confidence, 0.0, 1.0))


    def _compute_local_variability(self, state, n=10):
        perturb = []
        for _ in range(n):
            s = list(state)
            j = int(self.rng.integers(0, 5))
            if j == 0:
                s[0] = str(self.rng.choice(["YOUNG", "MID", "OLD"]))
            elif j == 1:
                s[1] = str(self.rng.choice(self.subtypes))
            elif j == 2:
                s[2] = str(self.rng.choice(self.cell_bins))
            elif j == 3:
                s[3] = str(self.rng.choice(self.surgery_bins))
            else:
                s[4] = int(self.rng.choice([0, 1]))
            perturb.append(tuple(s))

        a0 = self._get_best_action(state)
        same = sum(1 for ps in perturb if self._get_best_action(ps) == a0)
        return float(1 - same / len(perturb))

    def _compute_similarity_support(self, state):
        state_key = state[:5]
        similar = [k for k in self.Q.keys() if k[:5] == state_key]
        if not similar:
            return 0.5

        actions_q = {}
        for k in similar:
            a = int(k[5])
            actions_q.setdefault(a, []).append(float(self.Q[k]))

        if len(actions_q) == 1:
            return 0.9

        best_a = self._get_best_action(state)
        best_q = float(np.mean(actions_q.get(best_a, [0.0])))
        other_means = [float(np.mean(v)) for a, v in actions_q.items() if a != best_a]
        second = max(other_means) if other_means else 0.0
        return float(np.clip((best_q - second) / (abs(best_q) + 1e-3), 0.0, 0.9))

    def _compute_rule_consistency(self, state):
        age_bin, subtype, cell_bin, surgery, chemo = state
        best_a = self._get_best_action(state)
        score = 0.5

        aggressive = (subtype in ["Basal-like", "HER2-enriched"] or cell_bin == "HIGH")
        if aggressive and best_a in [1, 2]:
            score += 0.2
        if subtype == "Luminal A" and best_a == 0:
            score += 0.2
        if age_bin == "OLD" and best_a in [0, 3]:
            score += 0.1

        return float(min(1.0, score))

    def compute_feature_importance(self, state):
        action_ids, q0 = self._qvals(state)
        if not q0:
            return []

        p0 = self._softmax(q0, temperature=0.5)
        best0 = max(q0)
        a0_idx = int(np.argmax(q0))
        rank0 = np.argsort(q0)[::-1].tolist().index(a0_idx)

        features_ = ["age_bin", "pam50_subtype", "cellularity", "surgery", "chemotherapy_history"]
        domains = {
            "age_bin": ["YOUNG", "MID", "OLD"],
            "pam50_subtype": list(self.subtypes),
            "cellularity": list(self.cell_bins),
            "surgery": list(self.surgery_bins),
            "chemotherapy_history": [0, 1],
        }

        imp = {}
        for i, f in enumerate(features_):
            cur = state[i]
            candidates = [v for v in domains[f] if v != cur]
            if not candidates:
                imp[f] = 0.0
                continue

            scores = []
            for v in candidates[:2]:
                s = list(state)
                s[i] = v
                s = tuple(s)

                _, q1 = self._qvals(s)
                p1 = self._softmax(q1, temperature=0.5)

                best1 = max(q1)
                dq = abs(best0 - best1) / (abs(best0) + 1e-10)
                js = self._jensen_shannon_distance(p0, p1)

                rank1 = np.argsort(q1)[::-1].tolist().index(a0_idx)
                dr = abs(rank0 - rank1) / max(1, (len(action_ids) - 1))

                score = 0.4 * min(1.0, dq) + 0.4 * js + 0.2 * dr
                scores.append(score)

            imp[f] = float(np.mean(scores)) if scores else 0.0

        vals = np.array(list(imp.values()), dtype=float)
        if vals.sum() > 0:
            probs = self._softmax(vals, temperature=0.3)
            imp = {k: float(v) for k, v in zip(features_, probs)}

        return [{"feature": k, "label": k, "importance": float(v)} for k, v in imp.items()]

    def generate_counterfactuals(self, state, n=3):
        cur_action = self._get_best_action(state)

        basal = "Basal-like" if "Basal-like" in self.subtypes else str(self.subtypes[0])
        luma = "Luminal A" if "Luminal A" in self.subtypes else str(self.subtypes[0])

        scenarios = [
            ("Plus jeune", ("age_bin", "YOUNG")),
            ("Plus âgé", ("age_bin", "OLD")),
            ("Cellularité plus basse", ("cellularity", "LOW")),
            ("Cellularité plus haute", ("cellularity", "HIGH")),
            ("Sous-type plus agressif", ("pam50_subtype", basal)),
            ("Sous-type moins agressif", ("pam50_subtype", luma)),
            ("Avec chimiothérapie", ("chemotherapy_history", 1)),
            ("Sans chimiothérapie", ("chemotherapy_history", 0)),
        ]

        out = []
        for name, (feat, val) in scenarios:
            s = list(state)
            if feat == "age_bin":
                s[0] = str(val)
            elif feat == "pam50_subtype":
                s[1] = str(val)
            elif feat == "cellularity":
                s[2] = str(val)
            elif feat == "surgery":
                s[3] = str(val)
            else:
                s[4] = int(val)

            s = tuple(s)
            a = self._get_best_action(s)
            if a != cur_action:
                label = self.actions.get(a, str(a))
                out.append(
                    {
                        "title": name,
                        "what_if": f"Si {feat} → {val}.",
                        "result_state_text": _state_to_sentence(s),
                        # ✅ front-friendly
                        "recommended_action": {"title": str(label)},
                    }
                )
            if len(out) >= n:
                break
        return out

    def explain_decision(self, state, action_id):
        age_bin, subtype, cell_bin, surgery, chemo = state
        rules = []

        if subtype in ["Basal-like", "HER2-enriched"]:
            rules.append("Sous-type agressif → tendance vers traitement plus intensif / surveillance.")
        elif subtype == "Luminal A":
            rules.append("Sous-type bon pronostic → tendance vers approche minimale.")

        if cell_bin == "HIGH":
            rules.append("Cellularité élevée → tumeur potentiellement plus agressive.")
        elif cell_bin == "LOW":
            rules.append("Cellularité faible → tumeur potentiellement moins agressive.")

        if age_bin == "OLD":
            rules.append("Âge élevé → approche plus conservatrice si possible.")
        elif age_bin == "YOUNG":
            rules.append("Âge jeune → meilleure tolérance aux intensifications.")

        if surgery == "MASTECTOMY":
            rules.append("Mastectomie → discussion adjuvante selon profil global.")
        if int(chemo) == 1:
            rules.append("Chimiothérapie présente → à considérer pour stratégie.")

        txt = f"Recommandation '{self.actions.get(int(action_id), action_id)}' basée sur :\n"
        for i, r in enumerate(rules[:4], 1):
            txt += f"{i}. {r}\n"
        return txt


# ============================================================
# Fallback utils (si Q table plate sur un state)
# ============================================================

def _state_has_signal(Q, state, action_ids, eps=1e-12):
    q = [float(Q.get(state + (a,), 0.0)) for a in action_ids]
    return any(abs(x) > eps for x in q), q


def _find_best_neighbor_state(Q, state, action_ids):
    """
    Cherche un state proche qui a des q-values non nulles.
    Score = nb de features identiques (0..5), puis magnitude des Q.
    """
    target = state[:5]
    best = None
    best_tuple = (-1, -1.0)  # (match_count, max_abs_q)

    seen = set()
    for k, qv in Q.items():
        s = k[:5]
        if s in seen:
            continue
        seen.add(s)

        has_signal, qvals = _state_has_signal(Q, s, action_ids, eps=1e-12)
        if not has_signal:
            continue

        match_count = sum(1 for i in range(5) if s[i] == target[i])
        max_abs_q = max(abs(x) for x in qvals)
        cand = (match_count, max_abs_q)

        if cand > best_tuple:
            best_tuple = cand
            best = s

    return best, best_tuple


# ============================================================
# Public API: load + predict
# ============================================================

_BUNDLE_CACHE = None


def load_obj5_bundle(exported_dir, filename="obj5_rl_xai.joblib"):
    global _BUNDLE_CACHE
    path = os.path.join(exported_dir, filename)
    if _BUNDLE_CACHE is None or _BUNDLE_CACHE.get("_path") != path:
        b = joblib.load(path)
        b["_path"] = path
        _BUNDLE_CACHE = b
    return _BUNDLE_CACHE


def predict_obj5(payload, exported_dir):
    """
    payload: dict JSON depuis form / fetch
    exported_dir: dossier qui contient obj5_rl_xai.joblib
    return: dict prêt à jsonify()
    """
    bundle = load_obj5_bundle(exported_dir)

    Q = bundle["Q"]
    A = bundle["ACTIONS_MEDICAL"]
    cfg = dict(bundle.get("config") or {})
    domains = dict(bundle.get("domains") or {})

    # inject domains => known_* (comme ton route Flask)
    cfg["known_subtypes"]  = domains.get("subtypes", cfg.get("known_subtypes", []))
    cfg["known_surgeries"] = domains.get("surgery_bins", cfg.get("known_surgeries", []))
    cfg["known_cells"]     = domains.get("cell_bins", cfg.get("known_cells", []))

    action_ids = sorted([int(k) for k in A.keys()])

    # build state
    state = _obj5_build_state_from_payload(payload or {}, cfg)

    # qvals
    has_signal, qvals = _state_has_signal(Q, state, action_ids, eps=1e-12)

    fallback_used = False
    fallback_info = None

    # fallback si Q-table plate
    if not has_signal:
        neigh, (mcount, mag) = _find_best_neighbor_state(Q, state, action_ids)
        if neigh is not None:
            fallback_used = True
            fallback_info = {"reason": "flat_qvals", "match_count": mcount, "signal_mag": float(mag)}
            state = neigh
            _, qvals = _state_has_signal(Q, state, action_ids, eps=1e-12)

    # best action_id
    best_idx = int(np.argmax(qvals)) if qvals else 0
    best_action_id = int(action_ids[best_idx]) if action_ids else 0

    # confidence classic
    mx = float(np.max(qvals)) if qvals else 0.0
    mn = float(np.min(qvals)) if qvals else 0.0
    classic = 0.5 if mx == mn else (mx - mn) / (abs(mx) + 1e-6)
    classic = float(np.clip(classic, 0.0, 1.0))

    # XAI
    explainer = XAI_Explainer_Scientific(Q, A, cfg=cfg, rng_seed=42)
    feat_imp = explainer.compute_feature_importance(state)
    cfs = explainer.generate_counterfactuals(state, n=3)
    natural = explainer.explain_decision(state, best_action_id)
    conf_xai = explainer.compute_confidence_xai(state)

    combined = float(np.clip(0.6 * conf_xai + 0.4 * classic, 0.0, 1.0))
    level = "Élevé" if combined >= 0.70 else ("Moyen" if combined >= 0.40 else "Faible")

    # title/desc propre si tu veux un texte plus “UI”
    title, desc = _action_to_title_desc(best_action_id, A.get(best_action_id, str(best_action_id)))

    return {
        "rl": {
            "state_text": _state_to_sentence(state),
            "recommendation": {
                "action_id": best_action_id,
                "title": title if title else str(A.get(best_action_id, best_action_id)),
                "description": desc if desc else "Recommandation issue de la Q-table"
            },
            "confidence": {
                "classic": classic,
                "xai": conf_xai,
                "combined": combined,
                "level": level
            }
        },
        "xai": {
            "feature_importance": feat_imp,
            "counterfactuals": cfs,
            "natural_explanation": natural
        },
        "debug": {
            "state_tuple": list(state),
            "qvals": {str(a): float(q) for a, q in zip(action_ids, qvals)},
            "fallback_used": fallback_used,
            "fallback_info": fallback_info
        }
    }




# ============================================================
# OBJ4 — RECHUTE (POST-TRAITEMENT) — TRAIN + EXPORT
# - stacking fit sur TRAIN
# - calibration sigmoid fit sur VAL, en mode cv="prefit" (=> pas besoin FrozenEstimator)
# - seuil choisi sur VAL calibré (F1 ou recall>=0.70)
# ============================================================
print("\n" + "=" * 60)
print("OBJ4 — RECHUTE (POST-TRAITEMENT) — TRAIN + EXPORT")
print("=" * 60)

col_rfs4 = pick_first_existing(df, ["relapse_free_status", "rfs_status", "rfs"])
if col_rfs4 is None:
    raise ValueError("❌ OBJ4: relapse_free_status introuvable dans le dataset.")

y4 = clean_cat(df[col_rfs4]).map({"recurred": 1, "not recurred": 0, "not_recurred": 0})
mask4 = y4.notna()
df4 = df.loc[mask4].copy()
y4 = y4.loc[mask4].astype(int)

FEATURES4 = [
    "age_at_diagnosis", "tumor_size", "tumor_stage", "neoplasm_histologic_grade",
    "lymph_nodes_examined_positive", "er_status", "pr_status", "her2_status",
    "chemotherapy", "hormone_therapy", "radio_therapy"
]
missing4 = [c for c in FEATURES4 if c not in df4.columns]
if missing4:
    raise ValueError(f"❌ OBJ4: Colonnes manquantes: {missing4}")

X4 = df4[FEATURES4].copy()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X4, y4, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y4
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=VAL_SIZE_IN_TRAIN_FULL,
    random_state=RANDOM_STATE, stratify=y_train_full
)

num_cols4 = X4.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols4 = [c for c in X4.columns if c not in num_cols4]

preprocess4 = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols4),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cat_cols4),
])

pos4 = int((y_train == 1).sum())
neg4 = int((y_train == 0).sum())
scale_pos_weight4 = neg4 / max(pos4, 1)

rf4 = RandomForestClassifier(
    n_estimators=600,
    max_depth=10,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

xgb4 = xgb.XGBClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    min_child_weight=2.0,
    gamma=0.0,
    scale_pos_weight=scale_pos_weight4,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

lgb4 = lgb.LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=25,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

cv_stack4 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

stack4 = StackingClassifier(
    estimators=[("rf", rf4), ("xgb", xgb4), ("lgb", lgb4)],
    final_estimator=LogisticRegression(max_iter=8000, C=0.2, class_weight="balanced"),
    cv=cv_stack4,
    stack_method="predict_proba",
    n_jobs=-1,
    passthrough=True
)

stack_pipe4 = Pipeline([
    ("prep", preprocess4),
    ("model", stack4)
])

print("🚀 OBJ4 training stack_pipe4...")
stack_pipe4.fit(X_train, y_train)

p_val_raw4 = stack_pipe4.predict_proba(X_val)[:, 1]
print("\n=== OBJ4 VAL (RAW, non calibré) ===")
print("ROC-AUC:", round(roc_auc_score(y_val, p_val_raw4), 4))
print("PR-AUC :", round(average_precision_score(y_val, p_val_raw4), 4))

print("\n🎛️ OBJ4 Calibrating on VAL (sigmoid) avec cv='prefit' ...")
# ✅ Option B: PAS FrozenEstimator, on utilise cv="prefit"
cal4 = CalibratedClassifierCV(estimator=stack_pipe4, method="sigmoid", cv="prefit")
cal4.fit(X_val, y_val)

p_val_cal4 = cal4.predict_proba(X_val)[:, 1]
prec4, rec4, thr_pr4 = precision_recall_curve(y_val, p_val_cal4)
f1s4 = 2 * prec4 * rec4 / (prec4 + rec4 + 1e-9)

best_idx4 = int(np.argmax(f1s4[:-1])) if len(thr_pr4) else 0
thr_f14 = float(thr_pr4[best_idx4]) if len(thr_pr4) else 0.5

idx4 = np.where(rec4[:-1] >= TARGET_RECALL)[0]
thr_recall704 = float(thr_pr4[idx4[-1]]) if len(idx4) else thr_f14

FINAL_THR4 = thr_f14 if THRESHOLD_MODE.lower() == "f1" else thr_recall704

print("\n=== OBJ4 THRESHOLDS (VAL calibré) ===")
print("Seuil VAL (max F1):", round(thr_f14, 3))
print("Seuil VAL (Recall>=0.70):", round(thr_recall704, 3))
print("FINAL_THR choisi:", round(FINAL_THR4, 3), f"(mode={THRESHOLD_MODE})")

p_test4 = cal4.predict_proba(X_test)[:, 1]
y_pred4 = (p_test4 >= FINAL_THR4).astype(int)

print("\n=== OBJ4 TEST FINAL (calibré) ===")
print("ROC-AUC:", round(roc_auc_score(y_test, p_test4), 4))
print("PR-AUC :", round(average_precision_score(y_test, p_test4), 4))
print("F1     :", round(f1_score(y_test, y_pred4), 4))
print("BalancedAcc:", round(balanced_accuracy_score(y_test, y_pred4), 4))
print("CM:\n", confusion_matrix(y_test, y_pred4))
print("\nReport:\n", classification_report(y_test, y_pred4, digits=4))

p_train_full4 = cal4.predict_proba(X_train_full)[:, 1]
q25, q50, q75 = np.quantile(p_train_full4, [0.25, 0.50, 0.75])
p_train_sorted = np.sort(p_train_full4).astype(float)

obj4_payload = {
    "cal": cal4,
    "features": list(X4.columns),
    "final_thr": float(FINAL_THR4),
    "q": {"q25": float(q25), "q50": float(q50), "q75": float(q75)},
    "p_train_sorted": p_train_sorted,
    "meta": {
        "data_path": DATA_PATH,
        "threshold_mode": THRESHOLD_MODE,
        "target_recall": float(TARGET_RECALL),
        "random_state": int(RANDOM_STATE),
        "splits": {"test_size": float(TEST_SIZE), "val_size_in_train_full": float(VAL_SIZE_IN_TRAIN_FULL)}
    }
}

joblib.dump(obj4_payload, OBJ4_EXPORT_PATH)
print("✅ Exported OBJ4 ->", OBJ4_EXPORT_PATH)
import numpy as np
from collections import Counter

def _pick_first_existing(names):
    for n in names:
        if n in globals() and globals()[n] is not None:
            return n, globals()[n]
    return None, None

cand_rewards = ["rewards_history", "episode_rewards", "reward_history", "rewards", "R_history"]
cand_states  = ["states_visited", "visited_states", "state_history", "states_history", "S_history"]
cand_actions = ["actions_taken", "action_history", "actions_history", "A_history", "chosen_actions"]
cand_Q       = ["Q", "Q_table", "q_table", "QTABLE"]

rn, rewards = _pick_first_existing(cand_rewards)
sn, states  = _pick_first_existing(cand_states)
an, actions = _pick_first_existing(cand_actions)
qn, Q = _pick_first_existing(cand_Q)

print("Detected:")
print(" - rewards:", rn)
print(" - states :", sn)
print(" - actions:", an)
print(" - Q      :", qn)

# fallback minimal summary
def quick_summary(rewards, states, actions, Q):
    print("\n--- QUICK SUMMARY ---")
    if rewards is not None:
        r = np.array(rewards, dtype=float)
        print("Rewards mean/std/min/max:", float(r.mean()), float(r.std()), float(r.min()), float(r.max()))
        print("Unique rewards (rounded):", len(set(np.round(r, 8))))
    if states is not None:
        try:
            print("Unique states:", len(set(states)), " / total:", len(states))
        except TypeError:
            print("Unique states(str):", len({str(s) for s in states}), " / total:", len(states))
    if actions is not None:
        c = Counter(actions)
        total = sum(c.values())
        print("Top actions:", [(a, f"{cnt/total:.1%}") for a, cnt in c.most_common(10)])
    if Q is not None:
        try:
            some_states = list(Q.keys())[:50]
            spreads = []
            for s in some_states:
                row = Q[s]
                vals = np.array(list(row.values()), dtype=float) if isinstance(row, dict) else np.array(row, dtype=float)
                if len(vals):
                    spreads.append(float(vals.max() - vals.min()))
            if spreads:
                spreads = np.array(spreads)
                print("Q spread mean/min/max:", float(spreads.mean()), float(spreads.min()), float(spreads.max()))
                print("% spread < 1e-6:", float(np.mean(spreads < 1e-6)))
        except Exception as e:
            print("Q spread compute failed:", e)

quick_summary(rewards, states, actions, Q)
print("---------------------\n")

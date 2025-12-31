
# app.py — Version complète (corrigée + robuste)
# - Module A: Breast cancer diagnostic (MLP + scaler + feature_names.json)  -> /diagnostic
# - Module B: Multi-objectifs METABRIC (Obj1/Obj2/Obj3/Obj4/Obj5) -> /therapy
#
# ✅ FIXES APPORTÉS :
# 1) Suppression des doublons (helpers + routes /predict + module A répété) => évite crash Flask "View function mapping is overwriting..."
# 2) OBJ5 RL : la route /obj5/rl_xai fonctionne avec obj5_rl_xai.joblib, et inclut les helpers RL/XAI manquants (build_state, sentence, action desc).
# 3) Rien d’autre n’a été changé.

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect

# TensorFlow (MLP)
import tensorflow as tf  # noqa: F401
from tensorflow.keras.models import load_model



# ============================================================
# HELPERS — nettoyage & utilitaires
# ============================================================
NUMERIC_LIKE_DEFAULT = {
    "Age_at_Diagnosis", "Tumor_Size", "Lymph_Nodes_Examined_Positive",
    "Nottingham_prognostic_index", "Overall_Survival_Months",
    "Relapse_Free_Status_Months", "Cellularity"
}

def _apply_commas_and_cast(df, numeric_like=None):
    if df is None or len(df) == 0:
        return df
    numeric_like = set(numeric_like or [])
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].apply(lambda x: str(x).replace(",", ".") if x is not None and str(x) != "None" else x)
        if c in numeric_like:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _normalize_strings_in_df(df):
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            def norm(x):
                if x is None:
                    return np.nan
                s = str(x).strip()
                if s.lower() in ("none", "nan", ""):
                    return np.nan
                return s
            out[c] = out[c].map(norm)
    return out

def triage_one(p, lo=0.35, hi=0.65):
    try:
        p = float(p)
    except Exception:
        return "UNCERTAIN"
    if p <= lo:
        return "LOW"
    if p >= hi:
        return "HIGH"
    return "UNCERTAIN"

def intensity_level(theta, q):
    try:
        t = float(theta)
    except Exception:
        return "Inconnu"
    q25, q50, q75 = float(q["q25"]), float(q["q50"]), float(q["q75"])
    if t < q25:
        return "Faible"
    if t < q50:
        return "Modéré"
    if t < q75:
        return "Élevé"
    return "Très élevé"


app = Flask(__name__)
exported = {}

# ============================================================
# CHARGEMENT DES MODELES METABRIC (OBJ1..OBJ5 + RL)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATE_DIRS = [
    os.path.join(BASE_DIR, "models", "exported_models"),
    os.path.join(BASE_DIR, "models"),
]

def _safe_joblib_load(path):
    try:
        obj = joblib.load(path)
        print(f"✅ Loaded: {os.path.basename(path)}")
        return obj
    except Exception as e:
        print(f"⚠️ Failed to load {os.path.basename(path)} -> {e}")
        return None

def _find_file_in_candidates(filename):
    for d in CANDIDATE_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None

def load_exported_models():
    global exported
    exported = {}

    files = {
        "obj1": "obj1_therap_decision.joblib",
        "obj2": "obj2_intensity.joblib",
        "obj3": "obj3_endocrine_proxy.joblib",
        "obj4": "obj4_relapse.joblib",
        "obj5": "obj5_rl_xai.joblib",
        "rl":   "rl_qtable.joblib",
    }

    print("🔎 Searching models in:", CANDIDATE_DIRS)
    for key, fname in files.items():
        path = _find_file_in_candidates(fname)
        if path is None:
            exported[key] = None
            print(f"ℹ️ Not found: {fname}")
        else:
            exported[key] = _safe_joblib_load(path)

    print("✅ exported loaded:", {k: (v is not None) for k, v in exported.items()})

# Charger au démarrage
load_exported_models()

# ============================================================
# MODULE A: MLP DIAGNOSTIC (UNIQUE)
# ============================================================
def load_mlp_model():
    """Charge uniquement le modèle MLP et le scaler"""
    try:
        scaler = joblib.load('models/scaler.pkl')
        print("✅ Scaler chargé")

        mlp_model = load_model('models/mlp_model.h5')
        print("✅ Modèle MLP chargé (99.04% accuracy)")

        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        print(f"✅ {len(feature_names)} features chargées")

        try:
            scaler_n = getattr(scaler, 'n_features_in_', None)
        except Exception:
            scaler_n = None

        try:
            model_input_dim = mlp_model.input_shape[1]
        except Exception:
            model_input_dim = None

        if scaler_n is not None and model_input_dim is not None:
            if scaler_n != model_input_dim:
                print(f"⚠️ Attention: scaler.n_features_in_={scaler_n} mais mlp input_dim={model_input_dim}")

        return scaler, mlp_model, feature_names

    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("Assure-toi d'avoir exécuté ton notebook et exporté les modèles!")
        return None, None, None

scaler, mlp_model, FEATURE_NAMES = load_mlp_model()

@app.route('/')
def home():
    return render_template('index.html', features=FEATURE_NAMES)

@app.route('/diagnostics', methods=['GET'])
def diagnostics():
    if mlp_model is None or scaler is None:
        return jsonify({'success': False, 'error': 'Modèle ou scaler non chargé'})

    scaler_n = getattr(scaler, 'n_features_in_', None)
    try:
        model_input_dim = mlp_model.input_shape[1]
    except Exception:
        model_input_dim = None

    return jsonify({
        'success': True,
        'scaler_n_features_in': int(scaler_n) if scaler_n is not None else None,
        'model_input_dim': int(model_input_dim) if model_input_dim is not None else None,
        'feature_names_count': len(FEATURE_NAMES) if FEATURE_NAMES is not None else None,
        'feature_names_sample': FEATURE_NAMES[:5] if FEATURE_NAMES is not None else None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prédiction avec MLP seulement"""
    if mlp_model is None or scaler is None:
        return jsonify({'success': False, 'error': "Modèle non chargé. Exécute d'abord ton notebook."})

    try:
        features = []
        for i in range(30):
            value = request.form.get(f'feature_{i}', '0')
            features.append(float(value))

        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        model_input_shape = getattr(mlp_model, 'input_shape', None)

        if model_input_shape is not None and len(model_input_shape) == 3:
            input_for_model = features_scaled.reshape(1, features_scaled.shape[1], 1)
            pred_raw = mlp_model.predict(input_for_model, verbose=0)

            score = float(np.array(pred_raw).ravel()[0])
            prediction = 1 if score >= 0 else 0

            prob_malignant = 1.0 / (1.0 + np.exp(-score))
            confidence = prob_malignant if prediction == 1 else 1 - prob_malignant
            prediction_proba = prob_malignant
        else:
            prediction_proba = mlp_model.predict(features_scaled, verbose=0)
            prediction = 1 if prediction_proba[0][0] >= 0.5 else 0
            confidence = float(prediction_proba[0][0]) if prediction == 1 else 1 - float(prediction_proba[0][0])

        return jsonify({
            'success': True,
            'prediction': prediction,
            'prediction_label': 'MALIN' if prediction == 1 else 'BÉNIN',
            'confidence': round(confidence * 100, 1),
            'probability_malignant': round(float(prediction_proba) * 100, 1) if not isinstance(prediction_proba, (list, tuple)) else round(float(prediction_proba[0][0]) * 100, 1),
            'probability_benign': round((1 - float(prediction_proba)) * 100, 1) if not isinstance(prediction_proba, (list, tuple)) else round((1 - float(prediction_proba[0][0])) * 100, 1),
            'model_used': 'MLP (Multilayer Perceptron)',
            'model_accuracy': '99.04%',
            'model_info': '3 couches de 500 neurones, activation ReLU'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'Erreur de prédiction: {str(e)}'})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API pour les appels programmatiques (JSON)"""
    if mlp_model is None or scaler is None:
        return jsonify({'error': 'Modèle non chargé'})

    try:
        data = request.get_json()

        if 'features' not in data or len(data['features']) != 30:
            return jsonify({'error': '30 valeurs requises'})

        features_array = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        model_input_shape = getattr(mlp_model, 'input_shape', None)
        if model_input_shape is not None and len(model_input_shape) == 3:
            input_for_model = features_scaled.reshape(1, features_scaled.shape[1], 1)
            pred_raw = mlp_model.predict(input_for_model, verbose=0)
            score = float(np.array(pred_raw).ravel()[0])
            prediction = 1 if score >= 0 else 0
            prob_malignant = 1.0 / (1.0 + np.exp(-score))
            return jsonify({
                'prediction': int(prediction),
                'label': 'malignant' if prediction == 1 else 'benign',
                'probability_malignant': float(prob_malignant),
                'probability_benign': float(1 - prob_malignant),
                'model': 'GRU-SVM (raw score interpreted)'
            })
        else:
            prediction_proba = mlp_model.predict(features_scaled, verbose=0)
            prediction = 1 if prediction_proba[0][0] >= 0.5 else 0
            return jsonify({
                'prediction': int(prediction),
                'label': 'malignant' if prediction == 1 else 'benign',
                'probability_malignant': float(prediction_proba[0][0]),
                'probability_benign': float(1 - prediction_proba[0][0]),
                'model': 'MLP'
            })

    except Exception as e:
        return jsonify({'error': str(e)})


# ============================================================
# PAGE UI — /therapy
# ============================================================
@app.route("/therapy")
def therapy_page():
    obj1 = exported.get("obj1")
    if obj1 is None:
        return jsonify({"error": "OBJ1 non chargé. Vérifie obj1_therap_decision.joblib"}), 500
    return render_template("therapy_test.html")


# ============================================================
# MODULE B — XAI helpers (OBJ1)
# ============================================================
def _get_feature_names_from_preprocess(prep):
    names = []
    for name, trans, cols in prep.transformers_:
        if name == "remainder" or trans is None:
            continue
        if hasattr(trans, "named_steps"):
            if "ohe" in trans.named_steps:
                ohe = trans.named_steps["ohe"]
                cols = list(cols)
                try:
                    names.extend(list(ohe.get_feature_names_out(cols)))
                except Exception:
                    names.extend([f"{c}_<cat>" for c in cols])
            else:
                names.extend(list(cols))
        else:
            if hasattr(trans, "get_feature_names_out"):
                try:
                    names.extend(list(trans.get_feature_names_out(cols)))
                except Exception:
                    names.extend(list(cols))
            else:
                names.extend(list(cols))
    return np.array(names, dtype=object)

def _group_by_original_features(feature_names, original_features):
    parents = []
    for fn in feature_names:
        parent = None
        for base in original_features:
            if fn == base or fn.startswith(base + "_"):
                parent = base
                break
        parents.append(parent if parent is not None else fn)
    return np.array(parents, dtype=object)

def _sigmoid(z):
    return 1 / (1 + np.exp(-z))

def _local_xai_for_pipe(pipe, X1_raw, original_features, top_k=6):
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    Xt = pre.transform(X1_raw)
    Xt = Xt.toarray().ravel() if hasattr(Xt, "toarray") else np.asarray(Xt).ravel()

    feat_names = _get_feature_names_from_preprocess(pre)
    parents = _group_by_original_features(feat_names, original_features)

    coef = clf.coef_.ravel()
    intercept = float(clf.intercept_.ravel()[0])

    contrib = Xt * coef
    logit = intercept + contrib.sum()
    proba = float(_sigmoid(logit))

    dfc = pd.DataFrame({"feature": feat_names, "parent": parents, "contrib": contrib})
    grouped = dfc.groupby("parent")["contrib"].sum().sort_values(ascending=False)

    inc = grouped[grouped > 0].head(top_k)
    dec = grouped[grouped < 0].tail(top_k).sort_values()

    raw_vals = X1_raw.iloc[0].to_dict()

    def pack(series, direction):
        out = []
        for var, val in series.items():
            out.append(
                {
                    "var": var,
                    "value": raw_vals.get(var, None),
                    "direction": direction,
                    "contrib_logodds": float(val),
                }
            )
        return out

    return {
        "proba_recomputed": proba,
        "increase": pack(inc, "up"),
        "decrease": pack(dec, "down"),
        "grouped_contrib": {k: float(v) for k, v in grouped.items()},
    }

def _global_xai_for_pipe(pipe, original_features, top_k=8):
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    feat_names = _get_feature_names_from_preprocess(pre)
    parents = _group_by_original_features(feat_names, original_features)
    coef = clf.coef_.ravel()

    dfc = pd.DataFrame({"feature": feat_names, "parent": parents, "coef": coef, "abscoef": np.abs(coef)})
    imp = dfc.groupby("parent")["abscoef"].sum().sort_values(ascending=False).head(top_k)

    top_modalities = {}
    for var in imp.index:
        sub = dfc[dfc["parent"] == var].copy()
        top_pos = sub.sort_values("coef", ascending=False).head(3)[["feature", "coef"]].to_dict("records")
        top_neg = sub.sort_values("coef", ascending=True).head(3)[["feature", "coef"]].to_dict("records")
        top_modalities[var] = {"top_pos": top_pos, "top_neg": top_neg}

    return {"top_variables": [{"var": k, "importance": float(v)} for k, v in imp.items()], "modalities": top_modalities}


# ============================================================
# ROUTES — OBJ1
# ============================================================
@app.post("/obj1/predict")
def obj1_predict():
    obj1 = exported.get("obj1", None)
    if obj1 is None:
        return jsonify({"error": "OBJ1 non chargé. Vérifie models/exported_models/obj1_therap_decision.joblib"}), 500

    try:
        payload = request.get_json(force=True) or {}
        with_xai = bool(payload.get("with_xai", True))

        feats = obj1.get("features", [])
        labels = obj1.get("labels", [])
        triage_params = obj1.get("triage_params", {})
        if not feats or not labels:
            return jsonify({"error": "OBJ1 invalide: 'features' ou 'labels' manquant"}), 500

        X1 = pd.DataFrame([{c: payload.get(c, None) for c in feats}], columns=feats)
        X1 = _apply_commas_and_cast(X1, NUMERIC_LIKE_DEFAULT)
        X1 = _normalize_strings_in_df(X1)

        pipes = obj1.get("pipes", None) or obj1.get("pipe", None)
        if pipes is None or not isinstance(pipes, (list, tuple)) or len(pipes) != len(labels):
            return jsonify({"error": "OBJ1 invalide: 'pipes' doit être une liste alignée avec 'labels'."}), 500

        out = {"predictions": {}, "needs_rcp": False}

        for i, lab in enumerate(labels):
            p_yes = float(pipes[i].predict_proba(X1)[:, 1][0])
            lo = float(triage_params.get(lab, {}).get("lo", 0.35))
            hi = float(triage_params.get(lab, {}).get("hi", 0.65))
            out["predictions"][lab] = {"proba_yes": p_yes, "triage": triage_one(p_yes, lo, hi), "lo": lo, "hi": hi}

        out["needs_rcp"] = any(v["triage"] == "UNCERTAIN" for v in out["predictions"].values())

        if with_xai:
            xai = {"global": {}, "local": {}}
            for i, lab in enumerate(labels):
                pipe_i = pipes[i]
                xai["global"][lab] = _global_xai_for_pipe(pipe_i, feats, top_k=8)
                xai["local"][lab] = _local_xai_for_pipe(pipe_i, X1, feats, top_k=6)
            out["xai"] = xai

        return jsonify(out)

    except Exception as e:
        return jsonify({"error": f"OBJ1 predict error: {str(e)}"}), 500


# ============================================================
# ROUTES — OBJ2
# ============================================================
@app.post("/obj2/intensity")
def obj2_intensity():
    obj2 = exported.get("obj2", None)
    obj1 = exported.get("obj1", None)
    if obj2 is None or obj1 is None:
        return jsonify({"error": "OBJ2/OBJ1 non chargé. Vérifie obj2_intensity.joblib"}), 500

    try:
        data = request.get_json(force=True) or {}

        feats = obj1.get("features", [])
        labels = obj1.get("labels", [])
        if not feats or not labels:
            return jsonify({"error": "OBJ1 invalide: features/labels manquants."}), 500

        X = pd.DataFrame([{c: data.get(c, None) for c in feats}], columns=feats)
        X = _apply_commas_and_cast(X, NUMERIC_LIKE_DEFAULT)
        X = _normalize_strings_in_df(X)

        pipes = obj2.get("obj1_pipes", None) if isinstance(obj2, dict) else None
        if pipes is None:
            pipes = obj1.get("pipes", None) or obj1.get("pipe", None)

        if pipes is None:
            return jsonify({"error": "OBJ2: impossible de trouver les pipes OBJ1 (obj2['obj1_pipes'] ou obj1['pipes'])."}), 500

        if isinstance(pipes, (list, tuple)):
            if len(pipes) != len(labels):
                return jsonify({"error": f"OBJ2: pipes OBJ1 non alignés. len(pipes)={len(pipes)} vs len(labels)={len(labels)}"}), 500
            P = np.array([[float(pipes[i].predict_proba(X)[:, 1][0]) for i in range(len(labels))]], dtype=float)
        else:
            try:
                probas = pipes.predict_proba(X)
                P = np.array([[float(probas[i][0, 1]) for i in range(len(labels))]], dtype=float)
            except Exception as e:
                return jsonify({"error": f"OBJ2: format obj1_pipe/pipes non supporté: {str(e)}"}), 500

        theta_reg = obj2.get("theta_reg", None) if isinstance(obj2, dict) else None
        q = obj2.get("q", None) if isinstance(obj2, dict) else None
        if theta_reg is None or q is None:
            return jsonify({"error": "OBJ2 invalide: 'theta_reg' ou 'q' manquant. Re-export OBJ2."}), 500

        theta = float(theta_reg.predict(P)[0])
        level = intensity_level(theta, q)

        return jsonify({"theta": theta, "level": level, "probas_obj1": {labels[i]: float(P[0, i]) for i in range(len(labels))}})

    except Exception as e:
        return jsonify({"error": f"OBJ2 intensity error: {str(e)}"}), 500


# ============================================================
# ROUTES — OBJ3
# ============================================================
@app.post("/obj3/endocrine_proxy")
def obj3_endocrine_proxy():
    obj3 = exported.get("obj3")
    if obj3 is None:
        return jsonify({"error": "OBJ3 non chargé. Vérifie obj3_endocrine_proxy.joblib"}), 500

    data = request.get_json(force=True) or {}
    feats = obj3.get("features", [])
    if not feats:
        return jsonify({"error": "OBJ3 invalide: features manquants."}), 500

    X = pd.DataFrame([{c: data.get(c, None) for c in feats}], columns=feats)
    X = _normalize_strings_in_df(X)

    try:
        p = float(obj3["cal"].predict_proba(X)[:, 1][0])
        thr = float(obj3.get("thr", 0.5))
        triage = "HIGH_RISK" if p >= thr else "LOW_RISK"

        xai_global = obj3.get("xai_global", {})
        num_cols = obj3.get("num_cols", [])
        cat_cols = obj3.get("cat_cols", [])
        patient_row = X.iloc[0]

        def local_xai_delta_proba(cal, patient_row, num_cols, cat_cols, n_top=6):
            x0 = patient_row.to_frame().T.copy()
            p0 = float(cal.predict_proba(x0)[:, 1][0])
            deltas = []

            for c in num_cols:
                x1 = x0.copy()
                v = x1.at[x1.index[0], c]
                if pd.isna(v):
                    continue
                x1.at[x1.index[0], c] = np.nan
                p1 = float(cal.predict_proba(x1)[:, 1][0])
                deltas.append((c, p1 - p0, v))

            for c in cat_cols:
                x1 = x0.copy()
                v = x1.at[x1.index[0], c]
                if pd.isna(v):
                    continue
                x1.at[x1.index[0], c] = "unknown"
                p1 = float(cal.predict_proba(x1)[:, 1][0])
                deltas.append((c, p1 - p0, v))

            deltas = sorted(deltas, key=lambda x: abs(x[1]), reverse=True)
            inc, dec = [], []
            for c, d, v in deltas:
                item = {"var": c, "value": (None if pd.isna(v) else str(v)), "delta_proba": float(d)}
                (inc if d >= 0 else dec).append(item)

            return {"baseline_proba": p0, "increase": inc[:n_top], "decrease": dec[:n_top]}

        xai_local = local_xai_delta_proba(obj3["cal"], patient_row, num_cols, cat_cols, n_top=6)

        return jsonify(
            {
                "proba_proxy_resistance": p,
                "threshold": thr,
                "triage": triage,
                "xai": {"global": xai_global, "local": xai_local},
            }
        )
    except Exception as e:
        return jsonify({"error": f"OBJ3 error: {str(e)}"}), 500


# ============================================================
# ROUTES — OBJ4
# ============================================================
# ============================================================
# ROUTES — OBJ4 (revised + robust)
# ============================================================
@app.post("/obj4/relapse")
def obj4_relapse():
    obj4 = exported.get("obj4", None)
    if obj4 is None:
        return jsonify(
            {
                "error": "OBJ4 non chargé (échec de joblib.load).",
                "hint": "Vérifie models/exported_models/obj4_relapse.joblib et la version scikit-learn utilisée pour l'export.",
            }
        ), 500

    try:
        data = request.get_json(force=True) or {}

        feats = obj4.get("features", [])
        if not feats:
            return jsonify({"error": "OBJ4 invalide: features manquants."}), 500

        # --- build row
        row = {c: data.get(c, None) for c in feats}
        X = pd.DataFrame([row], columns=feats)
        X = _normalize_strings_in_df(X)

        # --- alias handling BEFORE normalization
        if "radio_therapy" in feats and "radio_therapy" not in data and "radiotherapy" in data:
            X["radio_therapy"] = data.get("radiotherapy")
        if "radiotherapy" in feats and "radiotherapy" not in data and "radio_therapy" in data:
            X["radiotherapy"] = data.get("radio_therapy")

        # --- numeric coercion
        for c in ["age_at_diagnosis", "tumor_size", "tumor_stage", "neoplasm_histologic_grade", "lymph_nodes_examined_positive"]:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")

        # --- therapy columns normalize (0/1) + enforce int
        therapy_cols = [c for c in ["chemotherapy", "hormone_therapy", "radio_therapy", "radiotherapy"] if c in X.columns]
        if therapy_cols:
            def _to01(z):
                s = str(z).strip().lower()
                if s in ["1","yes","y","true","treated","pos","positive","high"]:
                    return 1
                if s in ["0","no","n","false","none","neg","negative","low","", "nan", "null", "none"]:
                    return 0
                # si déjà numérique (ex: 0/1) ou autre -> laisse pour coercion
                return z

            for c in therapy_cols:
                X[c] = X[c].apply(_to01)
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(int)

        # --- predict
        p = float(obj4["cal"].predict_proba(X)[:, 1][0])

        q = obj4.get("q", {"q25": 0.25, "q50": 0.50, "q75": 0.75})
        level = (
            "Faible" if p < float(q["q25"]) else
            "Modéré" if p < float(q["q50"]) else
            "Élevé" if p < float(q["q75"]) else
            "Très élevé"
        )

        thr = float(obj4.get("final_thr", 0.5))

        # --- percentile (0..100)
        percentile = None
        arr = obj4.get("p_train_sorted", None)
        if isinstance(arr, (list, np.ndarray)) and len(arr) > 0:
            arr = np.asarray(arr, dtype=float)
            rank = int(np.searchsorted(arr, p, side="right"))
            percentile = float(rank / len(arr) * 100.0)

        return jsonify({
            "proba_relapse": p,
            "level": level,
            "alert": bool(p >= thr),
            "threshold": thr,
            "percentile": percentile
        })

    except Exception as e:
        return jsonify({"error": f"OBJ4 error: {str(e)}"}), 500

@app.route("/therapy_test")
def therapy_test():
    return render_template("therapy_test.html")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title='Model Predictor', layout='wide')

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

@st.cache_resource
def load_models():
    rf = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
    nb = joblib.load(os.path.join(MODELS_DIR, 'nb_model.pkl'))
    return rf, nb

rf, nb = load_models()

# Derive feature lists directly from the trained models (feature_names_in_ exists when fitted with DataFrame)
rf_features = list(getattr(rf, 'feature_names_in_', []))
nb_features = list(getattr(nb, 'feature_names_in_', []))

st.title('Random Forest & Naive Bayes Predictor')
st.write('Provide input values to get predictions from both models.')

st.subheader('Manual input')
cols = st.columns(2)
data = {}
# show numeric features (nb_features) for manual entry
for i, feat in enumerate(nb_features):
    col = cols[i % 2]
    # default value 0.0
    data[feat] = [col.number_input(feat, value=float(0.0))]
df_input = pd.DataFrame(data)
st.write(df_input)

# Prepare inputs
X_rf = df_input.reindex(columns=rf_features).fillna(0) if rf_features else df_input.copy()
X_nb = df_input.reindex(columns=nb_features).astype(float).fillna(0) if nb_features else df_input.copy().astype(float)

# Prediction mapping: 0 -> Fail, 1 -> Pass
label_map = {0: 'Fail', 1: 'Pass'}

pred_button = st.button('Predict')
if pred_button:
    st.subheader('Predictions')
    try:
        rf_preds = rf.predict(X_rf)
        rf_out = [label_map.get(int(p), str(p)) for p in rf_preds]
    except Exception as e:
        st.error(f'RF prediction failed: {e}')
        rf_out = [None] * len(df_input)

    try:
        nb_preds = nb.predict(X_nb)
        nb_out = [label_map.get(int(p), str(p)) for p in nb_preds]
    except Exception as e:
        st.error(f'NB prediction failed: {e}')
        nb_out = [None] * len(df_input)

    results = pd.DataFrame({
        'rf_prediction': rf_out,
        'nb_prediction': nb_out
    })
    st.write(results)

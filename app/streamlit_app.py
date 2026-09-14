import streamlit as st
import joblib
import pandas as pd

model = joblib.load('models/rf_model.pkl')
imputer = joblib.load('models/imputer.pkl')
feature_names = joblib.load('models/feature_names.pkl')

st.title("LabSense — CBC Differential Diagnosis")
st.write("Enter CBC values below to get a ranked differential diagnosis.")
st.warning("Educational/research prototype only — not for clinical use.")

st.header("Enter CBC Values")

RANGES = {
    'HGB': (3, 25, "g/dL"),
    'MCV': (40, 150, "fL"),
    'RBC': (1, 8, "million/µL"),
    'WBC': (0.5, 100, "x10^9/L"),
    'HCT': (10, 65, "%"),
    'PLT': (0, 1500, "x10^9/L"),
}

def validated_input(label, key, default):
    val = st.number_input(label, value=default, key=key)
    if key in RANGES:
        lo, hi, unit = RANGES[key]
        if not (lo <= val <= hi):
            st.caption(f"⚠️ Must be between {lo} and {hi} {unit}")
    return val

col1, col2 = st.columns(2)

inputs = {}
with col1:
    inputs['WBC'] = validated_input("WBC", 'WBC', 7.5)
    inputs['RBC'] = validated_input("RBC", 'RBC', 4.6)
    inputs['HGB'] = validated_input("Hemoglobin (HGB)", 'HGB', 12.0)
    inputs['HCT'] = validated_input("Hematocrit (HCT)", 'HCT', 46.0)
    inputs['MCV'] = validated_input("MCV", 'MCV', 86.0)
    inputs['MCH'] = st.number_input("MCH", value=32.0)
    inputs['MCHC'] = st.number_input("MCHC", value=31.8)

with col2:
    inputs['PLT'] = validated_input("Platelets (PLT)", 'PLT', 230.0)
    inputs['PDW'] = st.number_input("PDW", value=14.3)
    inputs['PCT'] = st.number_input("PCT", value=0.25)
    inputs['LYMp'] = st.number_input("Lymphocyte % (LYMp)", value=0.0)
    inputs['NEUTp'] = st.number_input("Neutrophil % (NEUTp)", value=0.0)
    inputs['LYMn'] = st.number_input("Lymphocyte count (LYMn)", value=0.0)
    inputs['NEUTn'] = st.number_input("Neutrophil count (NEUTn)", value=0.0)

has_error = any(
    not (RANGES[k][0] <= inputs[k] <= RANGES[k][1])
    for k in RANGES
)

if has_error:
    st.error("Please fix the highlighted fields above before submitting.")

if st.button("Get Differential Diagnosis", disabled=has_error):
    row = {}
    for col in feature_names:
        if col.endswith('_was_measured'):
            base_col = col.replace('_was_measured', '')
            row[col] = 1 if inputs.get(base_col, 0) != 0 else 0
        else:
            val = inputs.get(col, None)
            row[col] = None if val == 0.0 and col in ['LYMp','NEUTp','LYMn','NEUTn'] else val

    input_df = pd.DataFrame([row])[feature_names]
    input_imputed = imputer.transform(input_df)

    probs = model.predict_proba(input_imputed)[0]
    classes = model.classes_

    results = pd.DataFrame({'Diagnosis': classes, 'Probability': probs})
    results = results.sort_values('Probability', ascending=False)
    results['Probability'] = (results['Probability'] * 100).round(1).astype(str) + '%'

    st.header("Ranked Differential")
    st.table(results)

st.markdown("---")
st.caption("Built by Anshika Garg · dnamazing17x@gmail.com")
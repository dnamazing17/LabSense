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
col1, col2 = st.columns(2)

inputs = {}
with col1:
    inputs['WBC'] = st.number_input("WBC", value=7.5)
    inputs['RBC'] = st.number_input("RBC", value=4.6)
    inputs['HGB'] = st.number_input("Hemoglobin (HGB)", value=12.0)
    inputs['HCT'] = st.number_input("Hematocrit (HCT)", value=46.0)
    inputs['MCV'] = st.number_input("MCV", value=86.0)
    inputs['MCH'] = st.number_input("MCH", value=32.0)
    inputs['MCHC'] = st.number_input("MCHC", value=31.8)

with col2:
    inputs['PLT'] = st.number_input("Platelets (PLT)", value=230.0)
    inputs['PDW'] = st.number_input("PDW", value=14.3)
    inputs['PCT'] = st.number_input("PCT", value=0.25)
    inputs['LYMp'] = st.number_input("Lymphocyte % (LYMp)", value=0.0)
    inputs['NEUTp'] = st.number_input("Neutrophil % (NEUTp)", value=0.0)
    inputs['LYMn'] = st.number_input("Lymphocyte count (LYMn)", value=0.0)
    inputs['NEUTn'] = st.number_input("Neutrophil count (NEUTn)", value=0.0)

if st.button("Get Differential Diagnosis"):
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
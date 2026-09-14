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
    errors = []
    if not (3 <= inputs['HGB'] <= 25):
        errors.append("Hemoglobin (HGB) must be between 3 and 25 g/dL")
    if not (40 <= inputs['MCV'] <= 150):
        errors.append("MCV must be between 40 and 150 fL")
    if not (1 <= inputs['RBC'] <= 8):
        errors.append("RBC must be between 1 and 8 million/µL")
    if not (0.5 <= inputs['WBC'] <= 100):
        errors.append("WBC must be between 0.5 and 100 x10^9/L")
    if not (10 <= inputs['HCT'] <= 65):
        errors.append("HCT must be between 10 and 65%")
    if not (0 <= inputs['PLT'] <= 1500):
        errors.append("Platelets (PLT) must be between 0 and 1500 x10^9/L")
    if inputs['LYMp'] != 0 and not (0 <= inputs['LYMp'] <= 100):
        errors.append("Lymphocyte % must be between 0 and 100")
    if inputs['NEUTp'] != 0 and not (0 <= inputs['NEUTp'] <= 100):
        errors.append("Neutrophil % must be between 0 and 100")

    if errors:
        st.error("Please fix the following before submitting:\n\n" + "\n".join(f"- {e}" for e in errors))
    else:
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
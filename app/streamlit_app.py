import streamlit as st
import joblib
import pandas as pd

model = joblib.load('models/rf_model.pkl')
imputer = joblib.load('models/imputer.pkl')
feature_names = joblib.load('models/feature_names.pkl')

DIAGNOSIS_INFO = {
    'Healthy': {
        'description': 'CBC values fall within normal reference ranges, with no pattern suggesting a blood disorder.',
        'tests': 'None indicated based on CBC alone; routine follow-up as advised by a clinician.',
        'link': 'https://medlineplus.gov/lab-tests/complete-blood-count-cbc/'
    },
    'Iron deficiency anemia': {
        'description': 'A microcytic, hypochromic anemia caused by insufficient iron for hemoglobin production, often due to blood loss, poor intake, or malabsorption.',
        'tests': 'Serum ferritin, serum iron, TIBC/transferrin saturation, peripheral smear.',
        'link': 'https://medlineplus.gov/ency/article/000584.htm'
    },
    'Normocytic hypochromic anemia': {
        'description': 'Anemia with normal-sized red cells but reduced hemoglobin content, seen in early iron deficiency, chronic disease, or mixed nutritional deficiencies.',
        'tests': 'Ferritin, iron studies, inflammatory markers (CRP/ESR), reticulocyte count.',
        'link': 'https://medlineplus.gov/ency/article/000560.htm'
    },
    'Normocytic normochromic anemia': {
        'description': 'Anemia with normal-sized, normally colored red cells — commonly due to chronic disease, kidney disease, acute blood loss, or bone marrow suppression.',
        'tests': 'Reticulocyte count, kidney function tests, inflammatory markers, peripheral smear.',
        'link': 'https://medlineplus.gov/ency/article/000560.htm'
    },
    'Other microcytic anemia': {
        'description': 'Small red cells from a cause other than classic iron deficiency, such as thalassemia trait or anemia of chronic disease.',
        'tests': 'Hemoglobin electrophoresis, ferritin, iron studies, peripheral smear.',
        'link': 'https://medlineplus.gov/ency/article/000589.htm'
    },
    'Macrocytic anemia': {
        'description': 'Anemia with abnormally large red cells, often linked to vitamin B12 or folate deficiency, liver disease, or hypothyroidism.',
        'tests': 'Vitamin B12, folate levels, liver function tests, thyroid function tests, peripheral smear.',
        'link': 'https://medlineplus.gov/ency/article/000578.htm'
    },
    'Thrombocytopenia': {
        'description': 'A low platelet count, which can result from decreased production, increased destruction, or sequestration of platelets.',
        'tests': 'Peripheral smear, coagulation panel, liver function tests, viral serologies if indicated.',
        'link': 'https://medlineplus.gov/ency/article/000586.htm'
    },
    'Leukemia': {
        'description': 'A group of blood cancers marked by abnormal white blood cell production, which can crowd out normal blood cells.',
        'tests': 'Peripheral blood smear, bone marrow biopsy, flow cytometry — should be evaluated urgently by a specialist.',
        'link': 'https://medlineplus.gov/leukemia.html'
    },
    'Leukemia with thrombocytopenia': {
        'description': 'A leukemia pattern with an accompanying low platelet count, which can increase bleeding risk.',
        'tests': 'Peripheral blood smear, bone marrow biopsy, flow cytometry, coagulation panel — urgent specialist evaluation recommended.',
        'link': 'https://medlineplus.gov/leukemia.html'
    },
}

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

def validated_input(label, key, placeholder):
    val = st.number_input(label, value=None, placeholder=placeholder, key=key)
    if key in RANGES and val is not None:
        lo, hi, unit = RANGES[key]
        if not (lo <= val <= hi):
            st.caption(f"⚠️ Must be between {lo} and {hi} {unit}")
    return val

def optional_input(label, key, placeholder):
    return st.number_input(label, value=None, placeholder=placeholder, key=key)

col1, col2 = st.columns(2)

inputs = {}
with col1:
    inputs['WBC'] = validated_input("WBC", 'WBC', "e.g. 7.5")
    inputs['RBC'] = validated_input("RBC", 'RBC', "e.g. 4.6")
    inputs['HGB'] = validated_input("Hemoglobin (HGB)", 'HGB', "e.g. 12.0")
    inputs['HCT'] = validated_input("Hematocrit (HCT)", 'HCT', "e.g. 46.0")
    inputs['MCV'] = validated_input("MCV", 'MCV', "e.g. 86.0")
    inputs['MCH'] = optional_input("MCH", 'MCH', "e.g. 32.0")
    inputs['MCHC'] = optional_input("MCHC", 'MCHC', "e.g. 31.8")

with col2:
    inputs['PLT'] = validated_input("Platelets (PLT)", 'PLT', "e.g. 230.0")
    inputs['PDW'] = optional_input("PDW", 'PDW', "e.g. 14.3")
    inputs['PCT'] = optional_input("PCT", 'PCT', "e.g. 0.25")
    inputs['LYMp'] = optional_input("Lymphocyte % (LYMp)", 'LYMp', "optional")
    inputs['NEUTp'] = optional_input("Neutrophil % (NEUTp)", 'NEUTp', "optional")
    inputs['LYMn'] = optional_input("Lymphocyte count (LYMn)", 'LYMn', "optional")
    inputs['NEUTn'] = optional_input("Neutrophil count (NEUTn)", 'NEUTn', "optional")

required_fields = ['WBC', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT']
missing = [f for f in required_fields if inputs[f] is None]

has_range_error = any(
    inputs[k] is not None and not (RANGES[k][0] <= inputs[k] <= RANGES[k][1])
    for k in RANGES
)

if missing:
    st.info(f"Please fill in: {', '.join(missing)}")
if has_range_error:
    st.error("Please fix the highlighted fields above before submitting.")

if st.button("Get Differential Diagnosis", disabled=bool(missing) or has_range_error):
    row = {}
    for col in feature_names:
        if col.endswith('_was_measured'):
            base_col = col.replace('_was_measured', '')
            row[col] = 1 if inputs.get(base_col) not in (None, 0) else 0
        else:
            row[col] = inputs.get(col)

    input_df = pd.DataFrame([row])[feature_names]
    input_imputed = imputer.transform(input_df)

    probs = model.predict_proba(input_imputed)[0]
    classes = model.classes_

    results = pd.DataFrame({'Diagnosis': classes, 'Probability': probs})
    results = results.sort_values('Probability', ascending=False).reset_index(drop=True)
    results.index = results.index + 1
    results['Probability'] = (results['Probability'] * 100).round(1).astype(str) + '%'

    st.header("Ranked Differential")
    st.table(results)

        st.header("About Each Condition")
    st.caption("Follow-up tests are general educational guidance, not a prescription — always consult a clinician.")

    for diag in results['Diagnosis']:
        info = DIAGNOSIS_INFO.get(diag)
        if info:
            with st.expander(diag):
                st.write(info['description'])
                st.write(f"**Follow-up tests to confirm/rule out:** {info['tests']}")
                st.markdown(f"[Learn more]({info['link']})")st.markdown("---")
st.caption("Built by Anshika Garg · dnamazing17x@gmail.com")
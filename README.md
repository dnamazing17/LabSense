# \# LabSense

# 

# An explainable AI system for analyzing routine laboratory investigations and generating ranked disease differentials from multimodal clinical lab data.

# 

# ⚠️ \*\*Educational/research decision-support prototype — not a clinical diagnostic system.\*\*

# 

# \## What it does

# 

# LabSense combines multiple lab panels — CBC, KFT/RFT, LFT, TFT, glucose/diabetes markers, iron studies, electrolytes, lipid profile, inflammatory markers, and coagulation tests — to detect abnormal patterns and estimate a ranked differential of possible conditions, instead of relying on a single test in isolation.

# 

# \*\*Example:\*\*

Hb ↓, MCV ↓, MCH ↓, RDW ↑, Ferritin ↓, Serum iron ↓

↓

Microcytic hypochromic pattern

↓

Most likely: Iron deficiency anemia — 82%

Other possibilities: Thalassemia — 10%, Anemia of chronic disease — 6%



\## Pipeline



Lab results → preprocessing → abnormality detection → pattern recognition → ML model → ranked differential → explanation



\## Methodology highlights



\- Patient-level train/test splitting to prevent data leakage

\- Missingness handled as a signal, not just imputed away

\- Class-weighted handling of imbalance across conditions

\- Probability calibration (Platt/isotonic scaling)

\- SHAP-based explainability for every prediction



\## Tech stack



Python, scikit-learn, XGBoost, SHAP, Streamlit



\## Status



🚧 Active development — project structure set up, dataset sourcing and EDA in progress.



\## Disclaimer



This project is for educational and research purposes only. It is not validated for clinical use and should never be used to make real medical decisions.


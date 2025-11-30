# predict.py (updated with improved UI and translation guidance)
import streamlit as st
import datetime
from deep_translator import GoogleTranslator

# Compact Indian languages mapping (kept inline to avoid editing other files)
indian_languages = {
    "None": "",  # option to not translate
    "Hindi": "hi",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Bengali": "bn",
    "Urdu": "ur",
    "Odia": "or",
    "Assamese": "as"
}

# ---------------- existing helper functions (kept as-is) ----------------

def _load_crop_list():
    """
    Return a list of crop names.
    Replace this with a DB call (e.g. get_crops()) if you have such a function.
    """
    # Example static list — replace with dynamic retrieval if available
    # return ["Wheat", "Rice", "Maize", "Cotton", "Sugarcane","Apple"]
    return ["Rice","Wheat","Cotton","Bajra","Groundnut","Maize","Jowar","Castor","Tur (Pigeon Pea)","Moong (Green Gram)","Urad (Black Gram)","Sesame","Sugarcane","Potato","Onion","Cumin","Tobacco","Mustard","Rajma (Kidney Bean)","Chickpea","Soybean","Apple"]



def _run_advanced_predict(selected_crop):
    """
    Placeholder for advanced prediction logic.
    Replace with your model inference code.
    """
    # Example fake result; replace with real model call
    return {
        "mode": "advanced",
        "crop": selected_crop,
        "timestamp": st.session_state.get('_last_run_ts', ""),
        "prediction": f"Advanced predicted yield for {selected_crop}",
        "successRate": "90-95%",
        "season": "Kharif",
        "recommendedFertilizers": ["12 32 16", "Urea", "DAP"],
        "recommendedPesticides": ["Neem Oil", "Bt Toxin"],
        "recommendedIrrigation": "Weekly",
        "pestManagement": "Integrated Pest Management (IPM)",
        "notes": "Ensure proper soil testing before application."
    }   

    # return {
    #     "mode": "advanced",
    #     "crop": selected_crop,
    #     "timestamp": st.session_state.get('_last_run_ts', ""),
    #     "prediction": f"Advanced predicted yield for {selected_crop}",
    #     "successRate": "15-20%",
    #     "notes": "This crop not commonly grown in your region."
        
    # }
  


def _run_normal_predict():
    """
    Placeholder for normal (quick) prediction logic.
    Replace with your simple/default prediction code.
    """
    return {
        "mode": "normal",
        "timestamp": st.session_state.get('_last_run_ts', ""),
        "prediction": "Normal prediction result",
        "season": "Rabi",
        "recommendedCrops": "Wheat, Rice, Maize",
        "successRate": "80-85%",
        "recommendedFertilizers": ["0 0 50", "Urea", "DAP"],
        "recommendedPesticides": ["monocoto", "coragen"],
        "recommendedIrrigation": "Weekly",
        "pestManagement": "Integrated Pest Management (IPM)",
        "notes": "Ensure proper soil testing before application."
    }


# ---------------- small translation helpers ----------------

def _translate_text(text: str, target_code: str) -> str:
    """Translate text to the target language code. If target_code is empty, return original."""
    if not text or not target_code:
        return text
    try:
        return GoogleTranslator(source='auto', target=target_code).translate(text)
    except Exception as e:
        # don't crash the app if translation fails — return a helpful string
        return f"[Translation failed: {e}]"


# ---------------- main page function ----------------

def pridect_page(cookies=None):
    """
    Standalone Predict page. Call this from your main app when you want to switch to the Predict page.
    """

    # Page config and title
    st.set_page_config(page_title="Predict — Crop & Translate", page_icon="🌱", layout="wide")
    st.title("🌾 Predict Page")

    # Helpful guidance for users about translation
    st.info("Translations are performed on this Predict page using Deep Translator (Google) — select a language from the dropdown at the right to see translated fields below.")

    # Timestamp each run (store in session for demo)
    st.session_state['_last_run_ts'] = datetime.datetime.now().isoformat(timespec='seconds')

    # Top controls layout
    left, right = st.columns([3, 1])

    with left:
        st.subheader("Quick Predict")
        st.markdown("Use **Normal Predict** for a fast result or choose a crop on the right and select **Run Advanced** for a more detailed prediction.")

    with right:
        # Advanced predict dropdown (right-top)
        crop_list = _load_crop_list()
        selected_crop = st.selectbox("Advanced crop", crop_list, key="adv_crop")

        # Translation language dropdown (added) with helpful caption
        lang_choice = st.selectbox("Translate results to:", list(indian_languages.keys()), index=0, key="trans_lang")
        st.caption("Select a language to see translations of the human-facing fields below. 'None' keeps original language.")
        target_code = indian_languages.get(lang_choice, "")

        # Run advanced
        if st.button("Run Advanced"):
            try:
                result = _run_advanced_predict(selected_crop)
                st.session_state['prediction_result'] = result
                st.success(f"Advanced prediction run for: {selected_crop}")
            except Exception as e:
                st.error(f"Advanced prediction failed: {e}")

    # Middle area: Normal Predict button and quick info
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Normal Predict"):
            try:
                result = _run_normal_predict()
                st.session_state['prediction_result'] = result
                st.success("Normal prediction completed")
            except Exception as e:
                st.error(f"Normal prediction failed: {e}")
    with col2:
        st.write("**Tip:** After running a prediction, use the panel below to compare the original output and the translated text (if a language was selected).")

    st.markdown("---")

    # Show results if available
    if st.session_state.get('prediction_result'):
        res = st.session_state['prediction_result']

        # Summary header with metadata
        st.subheader("Last prediction result")
        meta_cols = st.columns(3)
        meta_cols[0].metric("Mode", res.get('mode', '—'))
        meta_cols[1].metric("Crop", res.get('crop', '—'))
        meta_cols[2].metric("Run time", res.get('timestamp', st.session_state.get('_last_run_ts', '—')))

        # Original JSON in an expander (kept for transparency)
        with st.expander("Show original JSON result", expanded=False):
            st.json(res)

        # Build translations only for human-facing fields
        # Fields to translate if present
        to_translate = {}
        for k in ['prediction', 'notes', 'pestManagement', 'recommendedIrrigation']:
            if k in res and isinstance(res[k], str) and res[k].strip():
                to_translate[k] = res[k]

        # Include list-like recommended fields (join them into a string)
        list_fields = ['recommendedFertilizers', 'recommendedPesticides', 'recommendedCrops']
        for k in list_fields:
            if k in res and isinstance(res[k], (list, tuple)) and res[k]:
                to_translate[k] = ", ".join(map(str, res[k]))
            elif k in res and isinstance(res[k], str) and res[k].strip():
                # sometimes recommendedCrops or others are strings; include them
                to_translate[k] = res[k]

        # Show results side-by-side for easy comparison
        if target_code:
            st.markdown(f"### Translated results — {lang_choice}")
            translated = {k: _translate_text(v, target_code) for k, v in to_translate.items()}

            # Two-column layout: original vs translated
            orig_col, trans_col = st.columns(2)
            orig_col.markdown("**Original (English or detected language)**")
            trans_col.markdown(f"**Translated ({lang_choice})**")

            for k in to_translate.keys():
                orig_col.write(f"- **{k}**: {to_translate[k]}")
                trans_col.write(f"- **{k}**: {translated.get(k, '')}")

        else:
            st.markdown("### Results (no translation selected)")
            # Show the human-facing fields neatly (only once)
            for k, v in to_translate.items():
                st.write(f"- **{k}**: {v}")

    # Back button to return to Home (main app listens for page state)
    st.markdown("---")
    if st.button("Back to Home"):
        st.session_state['page'] = "Home"
        st.rerun()



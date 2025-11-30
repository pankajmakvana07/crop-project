"""
Dynamic Predict Page - Fully integrated with database and ML models
"""
import streamlit as st
import datetime
from deep_translator import GoogleTranslator
from utils.model_integration import get_prediction_service

# Indian languages mapping
indian_languages = {
    "None": "",
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


def _translate_text(text: str, target_code: str) -> str:
    """Translate text to the target language code"""
    if not text or not target_code:
        return text
    try:
        return GoogleTranslator(source='auto', target=target_code).translate(text)
    except Exception as e:
        return f"[Translation failed: {e}]"


def _format_treatment_display(treatment, target_code=""):
    """Format treatment information for display with optional translation"""
    if not treatment:
        return "No treatment information available"
    
    lines = []
    
    # Pest Management
    if 'pestManagement' in treatment:
        label = _translate_text("Pest Management", target_code) if target_code else "Pest Management"
        value = _translate_text(treatment['pestManagement'], target_code) if target_code else treatment['pestManagement']
        lines.append(f"**{label}:** {value}")
    
    # Irrigation
    if 'recommendedIrrigation' in treatment:
        label = _translate_text("Recommended Irrigation", target_code) if target_code else "Recommended Irrigation"
        value = _translate_text(treatment['recommendedIrrigation'], target_code) if target_code else treatment['recommendedIrrigation']
        lines.append(f"**{label}:** {value}")
    
    # Fertilizers
    if 'recommendedFertilizers' in treatment:
        label = _translate_text("Recommended Fertilizers", target_code) if target_code else "Recommended Fertilizers"
        ferts = ", ".join(treatment['recommendedFertilizers'])
        lines.append(f"**{label}:** {ferts}")
    
    # Pesticides
    if 'recommendedPesticides' in treatment:
        label = _translate_text("Recommended Pesticides", target_code) if target_code else "Recommended Pesticides"
        pests = ", ".join(treatment['recommendedPesticides'])
        lines.append(f"**{label}:** {pests}")
    
    return "\n\n".join(lines)


def pridect_page(cookies=None):
    """
    Dynamic Predict page with full database and model integration
    """
    st.set_page_config(page_title="Crop Prediction", page_icon="🌱", layout="wide")
    st.title("🌾 Crop Prediction System")
    
    # Check if user is logged in
    if 'user_id' not in st.session_state:
        st.error("⚠️ Please login first to use the prediction system")
        if st.button("Go to Login"):
            st.session_state['page'] = "login"
            st.session_state['authenticated'] = False
            st.rerun()
        return
    
    user_id = st.session_state['user_id']
    
    # Initialize prediction service
    try:
        prediction_service = get_prediction_service()
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("Please run: `python model/Amodel.py` to train and generate model files")
        if st.button("⬅️ Back to Home"):
            st.session_state['page'] = "Home"
            st.rerun()
        return
    except Exception as e:
        st.error(f"❌ Failed to load prediction models: {e}")
        st.info("Please check model files in the 'model' directory")
        if st.button("⬅️ Back to Home"):
            st.session_state['page'] = "Home"
            st.rerun()
        return
    
    # Info banner
    st.info("🔬 This system uses AI models trained on Gujarat agricultural data to recommend crops based on your soil conditions")
    
    # Timestamp
    st.session_state['_last_run_ts'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Layout: Two columns
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.subheader("📊 Prediction Mode")
        st.markdown("""
        - **Normal Predict**: Get top 3 crop recommendations based on your soil data
        - **Advanced Predict**: Check if a specific crop can grow in your region
        """)
    
    with right_col:
        # Translation language selector
        lang_choice = st.selectbox(
            "🌐 Translate results to:",
            list(indian_languages.keys()),
            index=0,
            key="trans_lang"
        )
        target_code = indian_languages.get(lang_choice, "")
        
        if target_code:
            st.caption(f"Results will be translated to {lang_choice}")
    
    st.markdown("---")
    
    # Prediction controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌱 Normal Prediction")
        st.write("Get top 3 recommended crops for your region")
        
        if st.button("🚀 Run Normal Predict", use_container_width=True):
            with st.spinner("Analyzing your soil data and predicting crops..."):
                try:
                    result = prediction_service.predict_normal(user_id, top_k=3)
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                        st.info("Please add your soil details first from the 'Add Details' page")
                    else:
                        st.session_state['prediction_result'] = result
                        st.success("✅ Normal prediction completed successfully!")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
    
    with col2:
        st.subheader("🎯 Advanced Prediction")
        st.write("Check suitability for a specific crop")
        
        # Get available crops
        try:
            available_crops = prediction_service.get_available_crops()
            selected_crop = st.selectbox(
                "Select crop to analyze:",
                available_crops,
                key="adv_crop"
            )
            
            if st.button("🔍 Run Advanced Predict", use_container_width=True):
                with st.spinner(f"Analyzing {selected_crop} suitability..."):
                    try:
                        result = prediction_service.predict_advanced(user_id, selected_crop)
                        
                        if "error" in result:
                            st.error(f"❌ {result['error']}")
                            st.info("Please add your soil details first from the 'Add Details' page")
                        else:
                            st.session_state['prediction_result'] = result
                            st.success(f"✅ Advanced prediction for {selected_crop} completed!")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
        except Exception as e:
            st.error(f"Failed to load crop list: {e}")
    
    st.markdown("---")
    
    # Display results
    if st.session_state.get('prediction_result'):
        result = st.session_state['prediction_result']
        
        st.subheader("📋 Prediction Results")
        
        # Display user context
        context_cols = st.columns(4)
        context_cols[0].metric("District", result.get('district', 'N/A'))
        context_cols[1].metric("Taluka", result.get('taluka', 'N/A'))
        context_cols[2].metric("Soil Type", result.get('soil_type', 'N/A'))
        context_cols[3].metric("Soil pH", f"{result.get('soil_pH', 0):.1f}")
        
        st.markdown("---")
        
        # Display based on model type
        if result.get('model') == 'normal':
            st.subheader("🏆 Top 3 Recommended Crops")
            
            for idx, crop_data in enumerate(result.get('top_recommendations', []), 1):
                with st.expander(f"#{idx} - {crop_data['crop']} (Probability: {crop_data['probability']}%)", expanded=(idx==1)):
                    
                    # Metrics
                    metric_cols = st.columns(3)
                    metric_cols[0].metric(
                        "Suitability",
                        crop_data['suitability'],
                        delta=f"{crop_data['suitability_confidence']}% confidence"
                    )
                    metric_cols[1].metric(
                        "Predicted Yield",
                        f"{crop_data['predicted_yield_quintal_per_ha']:.1f}",
                        delta="Quintal/Ha"
                    )
                    metric_cols[2].metric(
                        "Recommendation Score",
                        f"{crop_data['probability']}%"
                    )
                    
                    # Treatment information
                    if crop_data.get('treatment'):
                        st.markdown("### 📝 Treatment Plan")
                        
                        if target_code:
                            col_orig, col_trans = st.columns(2)
                            with col_orig:
                                st.markdown("**Original (English)**")
                                st.markdown(_format_treatment_display(crop_data['treatment']))
                            with col_trans:
                                st.markdown(f"**Translated ({lang_choice})**")
                                st.markdown(_format_treatment_display(crop_data['treatment'], target_code))
                        else:
                            st.markdown(_format_treatment_display(crop_data['treatment']))
                    else:
                        st.warning("⚠️ This crop is not suitable for your region based on current conditions")
        
        elif result.get('model') == 'advanced':
            st.subheader(f"🎯 Analysis for {result.get('crop')}")
            
            # Prediction result
            prediction = result.get('prediction', 'Unknown')
            confidence = result.get('confidence', 0)
            
            if prediction == "Grow":
                st.success(f"✅ **{prediction}** - This crop CAN grow in your region (Confidence: {confidence}%)")
            else:
                st.error(f"❌ **{prediction}** - This crop may NOT grow well in your region (Confidence: {100-confidence}%)")
            
            # Reason
            reason = result.get('reason', '')
            if reason:
                st.markdown("### 💡 Analysis")
                if target_code:
                    col_orig, col_trans = st.columns(2)
                    with col_orig:
                        st.markdown("**Original:**")
                        st.info(reason)
                    with col_trans:
                        st.markdown(f"**Translated ({lang_choice}):**")
                        st.info(_translate_text(reason, target_code))
                else:
                    st.info(reason)
            
            # Metrics
            metric_cols = st.columns(2)
            metric_cols[0].metric("Confidence", f"{confidence}%")
            metric_cols[1].metric("Predicted Yield", f"{result.get('predicted_yield_quintal_per_ha', 0):.1f} Quintal/Ha")
            
            # Treatment information
            if result.get('treatment'):
                st.markdown("### 📝 Recommended Treatment Plan")
                
                if target_code:
                    col_orig, col_trans = st.columns(2)
                    with col_orig:
                        st.markdown("**Original (English)**")
                        st.markdown(_format_treatment_display(result['treatment']))
                    with col_trans:
                        st.markdown(f"**Translated ({lang_choice})**")
                        st.markdown(_format_treatment_display(result['treatment'], target_code))
                else:
                    st.markdown(_format_treatment_display(result['treatment']))
            else:
                st.warning("⚠️ No treatment plan available - crop not recommended for your region")
        
        # Raw JSON view
        with st.expander("🔍 View Raw JSON Output"):
            st.json(result)
    
    # Navigation
    st.markdown("---")
    nav_cols = st.columns(3)
    
    with nav_cols[0]:
        if st.button("⬅️ Back to Home", use_container_width=True):
            st.session_state['page'] = "app"
            st.rerun()
    
    with nav_cols[1]:
        if st.button("📝 Update Soil Details", use_container_width=True):
            st.session_state['page'] = "AddDetail"
            st.rerun()
    
    with nav_cols[2]:
        if st.button("🔄 Clear Results", use_container_width=True):
            if 'prediction_result' in st.session_state:
                del st.session_state['prediction_result']
            st.rerun()


if __name__ == "__main__":
    pridect_page()

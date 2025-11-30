import streamlit as st
import pandas as pd
import time
from utils.db_handler import save_user_detail, get_all_user_soil_details

# Load Gujarat village data
data = pd.read_csv(
    "page/taluka.csv",
    usecols=["District Name", "Taluka Name"],
    encoding='utf-8'
)

# =======================
# Custom CSS for Add Details Page
# =======================
def load_add_details_css():
    st.markdown("""
    <style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Override Streamlit's default text color */
    .main .block-container {
        color: #2c3e50;
    }
    
    /* Form container */
    .form-container {
        background: white;
        padding: 35px;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        margin: 20px 0;
        border-top: 5px solid #4caf50;
    }
    
    /* Page header */
    .page-header {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white !important;
        padding: 35px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .page-header h1 {
        margin: 0 !important;
        font-size: 2.5em;
        font-weight: 700;
        color: white !important;
    }
    
    .page-header p {
        margin: 10px 0 0 0 !important;
        font-size: 1.1em;
        opacity: 0.95;
        color: white !important;
    }
    
    /* Info card */
    .info-card {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    
    .info-card h3 {
        margin: 0 0 10px 0 !important;
        color: #2e7d32 !important;
    }
    
    .info-card p {
        color: #1b5e20 !important;
        line-height: 1.6;
        margin: 0;
    }
    
    /* History card */
    .history-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-left: 4px solid #ff9800;
    }
    
    .history-item {
        padding: 12px;
        background: #f5f5f5;
        border-radius: 8px;
        margin: 8px 0;
    }
    
    .history-label {
        font-weight: 600;
        color: #666 !important;
        display: inline-block;
        width: 120px;
    }
    
    .history-value {
        color: #333 !important;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Input field styling */
    .stSelectbox>div>div {
        border-radius: 10px;
    }
    
    .stNumberInput>div>div>input {
        border-radius: 10px;
    }
    
    /* Section divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #4caf50, transparent);
        margin: 35px 0;
    }
    
    /* pH indicator */
    .ph-indicator {
        background: linear-gradient(to right, #f44336, #ffeb3b, #4caf50);
        height: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    /* Headings */
    h3 {
        color: #2c3e50 !important;
        margin-top: 25px !important;
        margin-bottom: 15px !important;
    }
    
    /* Streamlit expander */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Streamlit info/success/warning boxes */
    .stAlert {
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Current details display box */
    .current-details-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .current-details-box p {
        margin: 8px 0 !important;
        color: #2c3e50 !important;
        font-size: 15px;
        line-height: 1.6;
    }
    
    .current-details-box strong {
        color: #555 !important;
        font-weight: 600;
        display: inline-block;
        min-width: 100px;
    }
    </style>
    """, unsafe_allow_html=True)


def add_detail_page():
    load_add_details_css()
    
    # Check if user is logged in
    if 'user_id' not in st.session_state:
        st.error("⚠️ You must login first to manage soil details!")
        if st.button("⬅ Back to Home"):
            st.session_state["page"] = "app"
            st.rerun()
        return
    
    # Check if user already has soil details (edit mode vs add mode)
    from utils.db_handler import get_user_soil_details, update_user_detail
    existing_details = get_user_soil_details(st.session_state['user_id'])
    is_edit_mode = existing_details is not None
    
    # Page Header
    header_title = "✏️ Edit Soil Details" if is_edit_mode else "🌍 Add Soil Details"
    header_subtitle = "Update your soil information" if is_edit_mode else "Add your soil information for accurate crop recommendations"
    
    st.markdown(f"""
    <div class="page-header">
        <h1 style="color: white;">{header_title}</h1>
        <p style="color: white;">{header_subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation buttons at top
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("⬅ Back to Home", use_container_width=True):
            st.session_state["page"] = "app"
            st.rerun()
    with col2:
        if st.button("🌾 Predict", use_container_width=True):
            st.session_state["page"] = "PredictPage"
            st.rerun()

    # Show mode indicator
    if is_edit_mode:
        st.info("✏️ **Edit Mode:** You are updating your existing soil details. You can only have one active soil record.")
    else:
        st.success("➕ **Add Mode:** You are adding your first soil details.")

    # Main Form Container
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    form_title = "✏️ Update Soil Information" if is_edit_mode else "📝 Enter Soil Information"
    st.markdown(f'<h3 style="color: #2c3e50; margin-top: 0;">{form_title}</h3>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-card">
        <h3>ℹ️ Why we need this information</h3>
        <p>Accurate soil data helps our AI model recommend the most suitable crops for your land, 
        maximizing yield and sustainability.</p>
    </div>
    """, unsafe_allow_html=True)

    # Get default values from existing details if in edit mode
    default_state = existing_details['state'] if is_edit_mode else "Gujarat"
    default_district_idx = 0
    default_taluka_idx = 0
    default_soil_type_idx = 0
    default_ph = existing_details['pH'] if is_edit_mode else 7.0
    
    # Prepare district list
    districts = sorted(data["District Name"].dropna().unique())
    
    if is_edit_mode and existing_details['district'] in districts:
        default_district_idx = districts.index(existing_details['district'])
    
    # Form fields in columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        state = st.selectbox("🗺️ State", ["Gujarat"], help="Select your state")
        district = st.selectbox(
            "📍 District", 
            districts, 
            index=default_district_idx,
            help="Select your district"
        )
        
    with col2:
        filtered_talukas = sorted(data[data["District Name"] == district]["Taluka Name"].dropna().unique())
        
        # Set default taluka index if in edit mode
        if is_edit_mode and existing_details['taluka'] in filtered_talukas:
            default_taluka_idx = filtered_talukas.index(existing_details['taluka'])
        
        taluka = st.selectbox(
            "🏘️ Taluka", 
            filtered_talukas, 
            index=default_taluka_idx,
            help="Select your taluka"
        )
        
        soil_types = ["Sandy", "Clay", "Loamy", "Silty", "Peaty", "Chalky"]
        if is_edit_mode and existing_details['soil_type'] in soil_types:
            default_soil_type_idx = soil_types.index(existing_details['soil_type'])
        
        soil_type = st.selectbox(
            "🌱 Soil Type", 
            soil_types,
            index=default_soil_type_idx,
            help="Select the predominant soil type in your area"
        )

    # pH input with visual indicator
    st.markdown('<h3 style="color: #2c3e50; margin-top: 20px;">🧪 Soil pH Level</h3>', unsafe_allow_html=True)
    ph = st.number_input(
        "Enter pH value (1.0 - 14.0)", 
        min_value=1.0, 
        max_value=14.0, 
        value=float(default_ph),
        step=0.1,
        help="pH 7 is neutral, <7 is acidic, >7 is alkaline"
    )
    
    # pH indicator
    st.markdown('<div class="ph-indicator"></div>', unsafe_allow_html=True)
    
    if ph < 6.0:
        st.info("🔴 Acidic soil - Some crops prefer acidic conditions")
    elif ph > 8.0:
        st.info("🔵 Alkaline soil - Certain crops thrive in alkaline conditions")
    else:
        st.success("🟢 Neutral soil - Ideal for most crops")

    st.markdown('</div>', unsafe_allow_html=True)

    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        button_text = "💾 Update Details" if is_edit_mode else "💾 Save Details"
        save_clicked = st.button(button_text, use_container_width=True, type="primary")

    if save_clicked:
        if is_edit_mode:
            # Update existing details
            success = update_user_detail(
                detail_id=existing_details['id'],
                user_id=st.session_state['user_id'],
                state=state,
                district=district,
                taluka=taluka,
                soil_type=soil_type,
                ph=ph
            )
            
            if success:
                st.success("✅ Soil details updated successfully!")
                st.balloons()
                st.info("💡 Your changes have been saved. You can now proceed to crop prediction!")
                time.sleep(2)
                st.rerun()
            else:
                st.error("❌ Failed to update soil details. Please check your database connection.")
        else:
            # Save new details
            success = save_user_detail(
                user_id=st.session_state['user_id'],
                state=state,
                district=district,
                taluka=taluka,
                soil_type=soil_type,
                ph=ph
            )

            if success:
                st.success("✅ Soil details saved successfully!")
                st.balloons()
                st.info("💡 You can now proceed to crop prediction!")
                time.sleep(2)
                st.rerun()
            else:
                st.error("❌ Failed to save soil details. Please check your database connection.")

    # Section divider
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Show current details if in edit mode
    if is_edit_mode:
        st.markdown('<h3 style="color: #2c3e50; margin-top: 30px;">📋 Current Soil Details</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="current-details-box">
                <p><strong>State:</strong> {existing_details['state']}</p>
                <p><strong>District:</strong> {existing_details['district']}</p>
                <p><strong>Taluka:</strong> {existing_details['taluka']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="current-details-box">
                <p><strong>Soil Type:</strong> {existing_details['soil_type']}</p>
                <p><strong>pH Level:</strong> {existing_details['pH']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.info("ℹ️ **Note:** You can only maintain one active soil record. Updating will replace your current details.")

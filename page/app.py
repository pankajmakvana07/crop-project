# main_app.py
import time
import streamlit as st
from utils.db_handler import get_users, get_user_soil_details, get_all_user_soil_details
from utils.init_session import reset_session

# Import the dynamic predict page
from page.pridect_dynamic import pridect_page

# =======================
# Custom CSS for Modern UI
# =======================
def load_custom_css():
    st.markdown("""
    <style>
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Override Streamlit's default text color */
    .main .block-container {
        color: #2c3e50;
    }
    
    /* Dashboard card styling */
    .dashboard-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-left: 5px solid #4caf50;
        transition: transform 0.2s;
        min-height: 150px;
    }
    
    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .dashboard-card h3 {
        color: #2c3e50 !important;
        margin-top: 0 !important;
        margin-bottom: 15px !important;
    }
    
    .dashboard-card p {
        color: #555 !important;
        line-height: 1.6;
        margin-bottom: 0;
    }
    
    /* Welcome header */
    .welcome-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 35px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .welcome-header h1 {
        margin: 0 !important;
        font-size: 2.5em;
        font-weight: 700;
        color: white !important;
    }
    
    .welcome-header p {
        margin: 10px 0 0 0 !important;
        font-size: 1.2em;
        opacity: 0.95;
        color: white !important;
    }
    
    /* Stats card */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .stat-card h2 {
        margin: 0 !important;
        font-size: 2.5em;
        font-weight: 700;
        color: white !important;
    }
    
    .stat-card p {
        margin: 10px 0 0 0 !important;
        font-size: 1em;
        opacity: 0.95;
        color: white !important;
    }
    
    /* Soil detail card */
    .soil-card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 4px solid #2196f3;
    }
    
    .soil-card-header {
        font-size: 1.3em;
        font-weight: 600;
        color: #2c3e50 !important;
        margin-bottom: 15px;
    }
    
    .soil-detail-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #ecf0f1;
    }
    
    .soil-detail-row:last-child {
        border-bottom: none;
    }
    
    .soil-detail-label {
        font-weight: 600;
        color: #7f8c8d !important;
        font-size: 1em;
    }
    
    .soil-detail-value {
        color: #2c3e50 !important;
        font-weight: 500;
        font-size: 1em;
    }
    
    /* Navigation buttons */
    .nav-button-container {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
    
    /* Action cards */
    .action-card {
        background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
        color: white !important;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .action-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    }
    
    .action-card h3 {
        margin: 0;
        font-size: 1.5em;
        color: white !important;
    }
    
    .action-card p {
        margin: 10px 0 0 0;
        opacity: 0.95;
        color: white !important;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
    }
    
    /* Info box */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 18px;
        border-radius: 8px;
        margin: 15px 0;
        color: #1565c0 !important;
    }
    
    .info-box strong {
        color: #0d47a1 !important;
    }
    
    /* Section divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #4caf50, transparent);
        margin: 35px 0;
    }
    
    /* Headings */
    h3 {
        color: #2c3e50 !important;
        margin-top: 25px !important;
        margin-bottom: 15px !important;
    }
    
    /* Streamlit info/success/warning boxes */
    .stAlert {
        border-radius: 10px;
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)


def app_page(cookies):
    """
    Main app page with modern card-based UI.
    Controls login/logout/Add Detail navigation and routes to predict page.
    """
    load_custom_css()

    # Ensure page key exists
    if 'page' not in st.session_state:
        st.session_state['page'] = "Home"

    # Route: Predict page (separate file renders only predict UI)
    if st.session_state.get('page') == "PredictPage":
        pridect_page(cookies)
        return

    # --- Navigation Bar ---
    st.markdown('<div class="nav-button-container">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
    
    with col1:
        if st.session_state.get('guest_mode'):
            if st.button("🔐 Login", use_container_width=True):
                cookies['auth_user'] = ""
                cookies.save()
                time.sleep(0.2)
                reset_session()
                st.rerun()
        else:
            if st.button("🚪 Logout", use_container_width=True):
                cookies['auth_user'] = ""
                cookies.save()
                time.sleep(0.2)
                reset_session()
                st.rerun()
    
    with col2:
        # Check if user has soil details to show appropriate button text
        button_text = "📝 Add Soil Details"
        if 'user_id' in st.session_state:
            from utils.db_handler import has_user_soil_details
            if has_user_soil_details(st.session_state['user_id']):
                button_text = "✏️ Edit Soil Details"
        
        if st.button(button_text, use_container_width=True):
            st.session_state['page'] = "AddDetail"
            st.rerun()
    
    with col3:
        if st.button("🌾 Crop Prediction", use_container_width=True):
            st.session_state['page'] = "PredictPage"
            st.rerun()
    
    with col4:
        if st.button("📜 History", use_container_width=True):
            st.session_state['page'] = "PredictionHistory"
            st.rerun()
    
    with col5:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state['page'] = "Home"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Welcome Header ---
    username = st.session_state.get('user', 'User')
    if st.session_state.get('guest_mode'):
        username = "Guest"
    
    st.markdown(f"""
    <div class="welcome-header">
        <h1 style="color: white;">🌱 Welcome, {username}!</h1>
        <p style="color: white;">Your Smart Crop Recommendation Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Dashboard Content ---
    
    # Statistics Section
    st.markdown('<h3 style="color: #2c3e50; margin-top: 20px;">📊 Dashboard Overview</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    try:
        users = get_users()
        total_users = len(users) if users else 0
    except Exception:
        total_users = 0
    
    # Get user's soil details count
    user_soil_count = 0
    if 'user_id' in st.session_state:
        try:
            all_details = get_all_user_soil_details(st.session_state['user_id'])
            user_soil_count = len(all_details) if all_details else 0
        except Exception:
            user_soil_count = 0
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h2>{total_users}</h2>
            <p>Total Users</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h2>{user_soil_count}</h2>
            <p>Your Soil Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h2>Gujarat</h2>
            <p>Active Region</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Recent Predictions Section ---
    if 'user_id' in st.session_state and not st.session_state.get('guest_mode'):
        st.markdown('<h3 style="color: #2c3e50; margin-top: 30px;">📊 Recent Predictions</h3>', unsafe_allow_html=True)
        
        try:
            from utils.db_handler import get_user_prediction_history
            recent_predictions = get_user_prediction_history(st.session_state['user_id'], limit=3)
            
            if recent_predictions:
                for pred in recent_predictions:
                    pred_type = pred['prediction_type'].upper()
                    crop = pred['predicted_crop']
                    confidence = pred['confidence']
                    created_at = pred['created_at'].strftime("%Y-%m-%d %H:%M") if hasattr(pred['created_at'], 'strftime') else str(pred['created_at'])
                    
                    st.markdown(f"""
                    <div class="soil-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="color: #2c3e50; font-size: 1.1em;">{crop}</strong>
                                <span style="color: #7f8c8d; margin-left: 10px;">({pred_type})</span>
                            </div>
                            <div style="text-align: right;">
                                <div style="color: #27ae60; font-weight: 600;">{confidence:.1f}% confidence</div>
                                <div style="color: #95a5a6; font-size: 0.9em;">{created_at}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if st.button("📜 View All Predictions", use_container_width=True):
                    st.session_state['page'] = "PredictionHistory"
                    st.rerun()
            else:
                st.info("📭 No predictions yet. Make your first prediction to see it here!")
        except Exception as e:
            st.warning(f"Could not load recent predictions: {e}")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # --- Your Soil Details Section ---
    if 'user_id' in st.session_state and not st.session_state.get('guest_mode'):
        st.markdown('<h3 style="color: #2c3e50; margin-top: 30px;">🌍 Your Soil Details</h3>', unsafe_allow_html=True)
        
        try:
            soil_details = get_user_soil_details(st.session_state['user_id'])
            
            if soil_details:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="soil-card">
                        <div class="soil-card-header">📍 Latest Soil Information</div>
                        <div class="soil-detail-row">
                            <span class="soil-detail-label">State:</span>
                            <span class="soil-detail-value">{soil_details['state']}</span>
                        </div>
                        <div class="soil-detail-row">
                            <span class="soil-detail-label">District:</span>
                            <span class="soil-detail-value">{soil_details['district']}</span>
                        </div>
                        <div class="soil-detail-row">
                            <span class="soil-detail-label">Taluka:</span>
                            <span class="soil-detail-value">{soil_details['taluka']}</span>
                        </div>
                        <div class="soil-detail-row">
                            <span class="soil-detail-label">Soil Type:</span>
                            <span class="soil-detail-value">{soil_details['soil_type']}</span>
                        </div>
                        <div class="soil-detail-row" style="border-bottom: none;">
                            <span class="soil-detail-label">pH Level:</span>
                            <span class="soil-detail-value">{soil_details['pH']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.info("✅ Soil details saved! Ready for crop prediction.")
                    if st.button("🔄 Update Details", use_container_width=True):
                        st.session_state['page'] = "AddDetail"
                        st.rerun()
            else:
                st.markdown("""
                <div class="info-box">
                    <strong>ℹ️ No soil details found</strong><br>
                    Add your soil details to get personalized crop recommendations!
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("➕ Add Your First Soil Detail", use_container_width=True):
                    st.session_state['page'] = "AddDetail"
                    st.rerun()
        
        except Exception as e:
            st.error(f"Error loading soil details: {e}")
    
    else:
        st.markdown("""
        <div class="info-box">
            <strong>👋 Guest Mode</strong><br>
            Login to save and view your soil details!
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Quick Actions ---
    st.markdown('<h3 style="color: #2c3e50; margin-top: 30px;">⚡ Quick Actions</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("""
            <div class="dashboard-card">
                <h3 style="color: #2c3e50; margin-top: 0;">🌾 Get Crop Recommendations</h3>
                <p style="color: #555;">Use our AI-powered model to get the best crop suggestions based on your soil conditions.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Start Prediction →", key="predict_action", use_container_width=True):
                st.session_state['page'] = "PredictPage"
                st.rerun()
    
    with col2:
        with st.container():
            st.markdown("""
            <div class="dashboard-card">
                <h3 style="color: #2c3e50; margin-top: 0;">📝 Manage Soil Data</h3>
                <p style="color: #555;">Add or update your soil information to get more accurate crop recommendations.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Manage Details →", key="details_action", use_container_width=True):
                st.session_state['page'] = "AddDetail"
                st.rerun()

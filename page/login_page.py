import streamlit as st
import time
import random
import string
from captcha.image import ImageCaptcha
from utils.db_handler import authenticate_user

# =======================
# Unified Custom CSS
# =======================
st.markdown("""
<style>
/* Main background gradient */
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* Override Streamlit defaults */
.main .block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* Welcome banner */
.auth-banner {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white !important;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 35px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    animation: fadeInDown 0.6s ease-out;
}

.auth-banner h2 {
    margin: 0 !important;
    font-size: 2.2em;
    font-weight: 700;
    color: white !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.auth-banner p {
    margin: 12px 0 0 0 !important;
    font-size: 1.1em;
    opacity: 0.95;
    color: white !important;
}

/* Auth card (login/signup) */
.auth-card {
    background: white;
    padding: 45px 40px;
    border-radius: 25px;
    box-shadow: 0 15px 50px rgba(0,0,0,0.35);
    border-top: 6px solid #4caf50;
    animation: fadeInUp 0.6s ease-out;
}

.auth-card h1 {
    font-size: 2em;
    font-weight: 700;
    color: #2c3e50 !important;
    margin: 0 0 10px 0 !important;
    text-align: center;
}

.auth-subtitle {
    color: #7f8c8d !important;
    font-size: 16px;
    margin-bottom: 30px !important;
    text-align: center;
}

/* Input fields */
.stTextInput>div>div>input, .stNumberInput>div>div>input {
    border: 2px solid #e0e0e0;
    border-radius: 12px;
    padding: 14px;
    font-size: 16px;
    transition: all 0.3s ease;
    background: #f8f9fa;
}

.stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
    border-color: #4caf50;
    box-shadow: 0 0 0 4px rgba(76, 175, 80, 0.1);
    background: white;
}

.stTextInput>div>div>input::placeholder {
    color: #adb5bd;
}

/* Input labels */
.stTextInput label, .stNumberInput label {
    color: #495057 !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    margin-bottom: 8px !important;
}

/* Checkbox */
.stCheckbox {
    padding: 12px 0;
}

.stCheckbox label {
    color: #495057 !important;
    font-size: 15px !important;
}

/* Buttons */
.stButton>button {
    border-radius: 12px;
    padding: 14px 28px;
    font-size: 16px;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}

.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
}

.stButton>button[kind="secondary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Captcha box */
.captcha-container {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border: 2px solid #dee2e6;
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.05);
}

.captcha-container p, .captcha-container label {
    color: #495057 !important;
}

.captcha-title {
    color: #495057 !important;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 12px;
    text-align: center;
}

/* Divider */
.auth-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, #dee2e6, transparent);
    margin: 25px 0;
}

/* Alert messages */
.stAlert {
    border-radius: 12px;
    padding: 16px;
    margin: 15px 0;
}

/* Animations */
@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Responsive */
@media (max-width: 768px) {
    .auth-card {
        padding: 30px 25px;
    }
    
    .auth-banner h2 {
        font-size: 1.8em;
    }
}
</style>
""", unsafe_allow_html=True)

# =======================
# Captcha Generator
# =======================
def generate_captcha():
    captcha_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    image = ImageCaptcha(width=280, height=90)
    data = image.generate(captcha_text)
    return captcha_text, data

# =======================
# Login Page
# =======================
def login_page(cookies, guest_mode=False):
    if "captcha_text" not in st.session_state:
        st.session_state["captcha_text"], st.session_state["captcha_data"] = generate_captcha()

    # Welcome banner
    st.markdown("""
    <div class="auth-banner">
        <h2>🌾 Smart Crop Recommendation System</h2>
        <p>Login to access personalized crop predictions and insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main login card
    col1, col2, col3 = st.columns([1, 2.5, 1])
    with col2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        
        # Header
        st.markdown("<h1>🔐 Welcome Back</h1>", unsafe_allow_html=True)
        st.markdown('<p class="auth-subtitle">Sign in to continue to your dashboard</p>', unsafe_allow_html=True)

        # Input fields
        user_input = st.text_input(
            "📧 Email or Phone Number",
            placeholder="Enter your email or phone number",
            key="login_user_input"
        )
        
        password = st.text_input(
            "🔒 Password",
            type="password",
            placeholder="Enter your password",
            key="login_password"
        )
        
        # Remember me checkbox
        remember_me = st.checkbox("Remember me", value=True)

        # Captcha section
        st.markdown('<div class="captcha-container">', unsafe_allow_html=True)
        st.markdown('<p class="captcha-title" style="color: #495057;">🔐 Security Verification</p>', unsafe_allow_html=True)
        
        col_cap1, col_cap2 = st.columns([4, 1])
        with col_cap1:
            st.image(st.session_state["captcha_data"], use_container_width=True)
        with col_cap2:
            refresh_clicked = st.button("🔄", key="refresh", help="Refresh Captcha", use_container_width=True)
        
        captcha_input = st.text_input(
            "Enter the characters shown above",
            placeholder="Enter captcha code",
            key="captcha_input"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Divider
        st.markdown('<div class="auth-divider"></div>', unsafe_allow_html=True)

        # Action buttons
        col_login, col_signup = st.columns(2)
        with col_login:
            login_clicked = st.button("🚀 Login", key="login", use_container_width=True, type="primary")
        with col_signup:
            signup_clicked = st.button("📝 Sign Up", key="signup", use_container_width=True, type="secondary")
        
        # Guest mode button
        if guest_mode:
            st.markdown('<div class="auth-divider"></div>', unsafe_allow_html=True)
            guest_clicked = st.button("👤 Continue as Guest", use_container_width=True)
        else:
            guest_clicked = None

        # Handle login
        if login_clicked:
            if not (user_input and password):
                st.error("⚠️ Please provide both email/phone and password")
            elif not captcha_input:
                st.error("⚠️ Please enter the captcha code")
            elif captcha_input.strip().upper() != st.session_state["captcha_text"]:
                st.error("❌ Incorrect captcha. Please try again.")
                # Refresh captcha for next attempt
                time.sleep(1.5)  # Give user time to see the error
                st.session_state["captcha_text"], st.session_state["captcha_data"] = generate_captcha()
                st.rerun()
            else:
                with st.spinner("🔄 Authenticating..."):
                    authenticated, user_id = authenticate_user(user_input, password)
                    if authenticated:
                        st.success("✅ Login successful! Redirecting...")
                        st.session_state['authenticated'] = True
                        st.session_state['user'] = user_input
                        st.session_state['user_id'] = user_id
                        st.session_state['guest_mode'] = False
                        st.session_state['page'] = 'app'
                        cookies['auth_user'] = user_input
                        cookies.save()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please check your email/phone and password.")
                        # Refresh captcha after failed login
                        st.session_state["captcha_text"], st.session_state["captcha_data"] = generate_captcha()

        # Handle captcha refresh
        if refresh_clicked:
            st.session_state["captcha_text"], st.session_state["captcha_data"] = generate_captcha()
            st.rerun()
        
        # Handle signup navigation
        if signup_clicked:
            st.session_state['page'] = 'signup'
            st.rerun()
        
        # Handle guest mode
        if guest_mode and guest_clicked:
            st.session_state['guest_mode'] = True
            st.session_state['authenticated'] = True
            st.session_state['page'] = 'app'
            cookies['auth_user'] = "guest"
            cookies.save()
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

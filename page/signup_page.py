import streamlit as st
import re
from utils.otp_handler import generate_otp, send_email
from utils.db_handler import save_user, verify_duplicate_user
import time

# =======================
# Unified Custom CSS (Same as Login)
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
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
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
    border-top: 6px solid #9c27b0;
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

/* OTP card */
.otp-card {
    background: white;
    padding: 45px 40px;
    border-radius: 25px;
    box-shadow: 0 15px 50px rgba(0,0,0,0.35);
    text-align: center;
    border-top: 6px solid #ff9800;
    animation: fadeInUp 0.6s ease-out;
}

.otp-card h3 {
    color: #2c3e50 !important;
    margin-bottom: 20px !important;
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
    border-color: #9c27b0;
    box-shadow: 0 0 0 4px rgba(156, 39, 176, 0.1);
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
    background: linear-gradient(135deg, #9c27b0 0%, #8e24aa 100%);
}

.stButton>button[kind="secondary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Section headers */
.section-header {
    color: #495057 !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    margin: 25px 0 15px 0 !important;
    padding-bottom: 10px;
    border-bottom: 2px solid #e9ecef;
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

/* Info box */
.info-highlight {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    border-left: 4px solid #2196f3;
    padding: 18px;
    border-radius: 12px;
    margin: 20px 0;
    color: #1565c0 !important;
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
    .auth-card, .otp-card {
        padding: 30px 25px;
    }
    
    .auth-banner h2 {
        font-size: 1.8em;
    }
}
</style>
""", unsafe_allow_html=True)

def is_valid_email(email):
    """Check if the provided email is valid using regex."""
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

def input_field(input_param, type):
    """Render an input field based on the type and store the value in session state."""
    if type == 'text':
        st.session_state[input_param] = st.text_input(
            f"👤 {input_param}",
            placeholder=f"Enter your {input_param.lower()}",
            key=f"signup_{input_param}"
        )
    elif type == 'number':
        st.session_state[input_param] = st.number_input(
            f"📱 {input_param}",
            step=1,
            format="%d",
            key=f"signup_{input_param}"
        )

def verifyOTP(otp_input):
    """Verify the OTP input by the user."""
    if otp_input == st.session_state['otp']:
        st.success("✅ OTP verified successfully!")
        time.sleep(1)
        st.session_state['verifying'] = False
        st.session_state['otp'] = ""
        # Save user and get user_id
        user_id = save_user(st.session_state['email'], st.session_state['password'], st.session_state['extra_input_params'])
        st.session_state['user_id'] = user_id  # Store user_id in session
        st.session_state['page'] = 'login'
        st.rerun()
    else:
        st.error("❌ Invalid OTP. Please try again.")

def signup_page(extra_input_params=False, confirmPass=False):
    """Render the signup page with optional extra input parameters and password confirmation."""
    
    # Welcome banner
    st.markdown("""
    <div class="auth-banner">
        <h2>🌱 Join Our Community</h2>
        <p>Create an account to get personalized crop recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.get('verifying', False):
        # Check if the user already exists
        if verify_duplicate_user(st.session_state['email']):
            st.error("⚠️ User already exists with this email!")
            time.sleep(1)
            st.session_state['verifying'] = False
            st.rerun()
        
        # OTP Verification UI
        col1, col2, col3 = st.columns([1, 2.5, 1])
        with col2:
            st.markdown('<div class="otp-card">', unsafe_allow_html=True)
            
            st.markdown("### 📧 Email Verification")
            
            st.markdown(f"""
            <div class="info-highlight">
                <p style="margin: 0; color: #1565c0;"><strong>📨 OTP Sent!</strong></p>
                <p style="margin: 8px 0 0 0; color: #1976d2;">
                    We've sent a 4-digit code to <strong>{st.session_state['email']}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("Please check your inbox and enter the verification code below.")
            
            print(st.session_state['otp'])
            if st.session_state['otp'] == "":
                st.session_state['otp'] = generate_otp()
                print(st.session_state['otp'])
                send_email(st.session_state['email'], st.session_state['otp'])
            
            otp_input = st.text_input(
                "🔢 Verification Code",
                placeholder="Enter 4-digit code",
                max_chars=6,
                key="otp_input"
            )
            
            st.markdown('<div class="auth-divider"></div>', unsafe_allow_html=True)
            
            col_verify, col_resend = st.columns(2)
            with col_verify:
                if st.button("✅ Verify Code", use_container_width=True, type="primary"):
                    if otp_input:
                        with st.spinner("🔄 Verifying..."):
                            verifyOTP(otp_input)
                    else:
                        st.warning("⚠️ Please enter the verification code")
            
            with col_resend:
                if st.button("🔄 Resend Code", use_container_width=True):
                    with st.spinner("📨 Sending new code..."):
                        st.session_state['otp'] = generate_otp()
                        send_email(st.session_state['email'], st.session_state['otp'])
                        time.sleep(0.5)
                        st.success("✅ New code sent to your email!")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        # Signup Form UI
        col1, col2, col3 = st.columns([1, 2.5, 1])
        with col2:
            st.markdown('<div class="auth-card">', unsafe_allow_html=True)
            
            # Back button
            if st.button("⬅ Back to Login", use_container_width=True):
                st.session_state['page'] = 'login'
                st.rerun()
            
            # Header
            st.markdown("<h1>📝 Create Account</h1>", unsafe_allow_html=True)
            st.markdown('<p class="auth-subtitle">Fill in your details to get started</p>', unsafe_allow_html=True)
            
            st.markdown('<div class="auth-divider"></div>', unsafe_allow_html=True)
            
            # Account Information Section
            st.markdown('<p class="section-header">🔐 Account Information</p>', unsafe_allow_html=True)
            
            # Email input with validation
            st.session_state['email'] = st.text_input(
                "📧 Email Address",
                placeholder="your.email@example.com",
                key="signup_email"
            )
            if st.session_state['email'] and not is_valid_email(st.session_state['email']):
                st.error("⚠️ Please enter a valid email address")

            # Password input
            st.session_state['password'] = st.text_input(
                "🔒 Password",
                type='password',
                placeholder="Create a strong password",
                key="signup_password"
            )
            
            # Confirm password if required
            confirm_password = ""
            if confirmPass:
                confirm_password = st.text_input(
                    "🔒 Confirm Password",
                    type='password',
                    placeholder="Re-enter your password",
                    key="signup_confirm_password"
                )
            
            # Extra input fields if any
            if extra_input_params:
                st.markdown('<div class="auth-divider"></div>', unsafe_allow_html=True)
                st.markdown('<p class="section-header">👤 Personal Information</p>', unsafe_allow_html=True)
                
                for input_param, type in st.session_state['extra_input_params'].items():
                    input_field(input_param, type)
            
            st.markdown('<div class="auth-divider"></div>', unsafe_allow_html=True)
            
            # Validate all required fields before proceeding
            can_register = st.session_state.get('email', '') and st.session_state.get('password', '')
            
            if confirmPass:
                can_register = can_register and (st.session_state.get('password', '') == confirm_password)
            
            if extra_input_params:
                can_register = can_register and all(st.session_state.get(param) for param in st.session_state['extra_input_params'])
            
            # Show validation errors
            if not can_register:
                if confirmPass and st.session_state.get('password') and confirm_password and st.session_state['password'] != confirm_password:
                    st.error("❌ Passwords do not match")
                elif not st.session_state.get('email'):
                    st.info("ℹ️ Please enter your email address")
                elif not st.session_state.get('password'):
                    st.info("ℹ️ Please create a password")
                elif extra_input_params and not all(st.session_state.get(param) for param in st.session_state['extra_input_params']):
                    st.info("ℹ️ Please fill in all required fields")
            
            # Register button
            if st.button("🚀 Create Account", use_container_width=True, type="primary"):
                if can_register:
                    with st.spinner("🔄 Creating your account..."):
                        st.session_state['verifying'] = True
                        time.sleep(0.5)
                        st.rerun()
                else:
                    if not st.session_state.get('email'):
                        st.error("⚠️ Please enter your email")
                    elif not st.session_state.get('password'):
                        st.error("⚠️ Please enter a password")
                    elif extra_input_params and not all(st.session_state.get(param) for param in st.session_state['extra_input_params']):
                        st.error("⚠️ Please fill in all required fields")
            
            st.markdown('</div>', unsafe_allow_html=True)

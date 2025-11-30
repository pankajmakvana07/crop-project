import time
import streamlit as st
from streamlit_cookies_manager import CookieManager

from page.login_page import login_page
from page.signup_page import signup_page
from page.app import app_page
from utils.init_session import init_session
from page.AddDetails import add_detail_page
from page.prediction_history import prediction_history_page

# ---------------- Cookies ----------------
cookies = CookieManager(prefix="myapp/")

# Wait for cookie manager to initialize
if not cookies.ready():
    st.stop()

# ---------------- Init Session ----------------
init_session()

# Restore login from cookie if available
if not st.session_state.get("authenticated", False):
    user_cookie = cookies.get("auth_user")

    user_val = None
    if isinstance(user_cookie, dict):
        user_val = user_cookie.get("value") or ""
    else:
        user_val = user_cookie or ""

    if isinstance(user_val, str) and user_val.strip() != "":
        st.session_state["authenticated"] = True
        st.session_state["user"] = user_val
        st.session_state["page"] = "app"

# ---------------- Extra inputs ----------------
st.session_state['extra_input_params'] = {
    'Username': 'text',
    'PhoneNumber': 'number',
    # 'Semester': 'number',
}
for input_param in st.session_state['extra_input_params'].keys():
    if input_param not in st.session_state:
        st.session_state[input_param] = None

# ---------------- Routing ----------------
if st.session_state.get('authenticated'):
    # Handle AddDetail page separately
    if st.session_state.get('page') == "AddDetail":
        add_detail_page()
    elif st.session_state.get('page') == "PredictionHistory":
        prediction_history_page(cookies)
    else:
        app_page(cookies)
else:
    if st.session_state.get('page') == 'login':
        login_page(cookies, guest_mode=True)
    elif st.session_state.get('page') == 'signup':
        signup_page(extra_input_params=True, confirmPass=True)

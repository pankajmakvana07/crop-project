#  pip install deep-translator

# from deep_translator import GoogleTranslator
# text = "Hello, how are you?"
# translated = GoogleTranslator(source='auto', target='hi').translate(text)
# print(translated)
# GoogleTranslator.get_supported_languages(as_dict=True)




import streamlit as st
from deep_translator import GoogleTranslator

# List of major Indian languages supported by Google Translate
indian_languages = {
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

# Streamlit App UI
st.set_page_config(page_title="Indian Language Translator", page_icon="🌐", layout="centered")
st.title("🌐 English to Indian Language Translator")

# Input text
text = st.text_area("Enter text in English (or any language):", height=150)

# Language selection
target_lang = st.selectbox("Select Target Language:", list(indian_languages.keys()))

# Translate button
if st.button("Translate"):
    if text.strip():
        try:
            translated_text = GoogleTranslator(source='auto', target=indian_languages[target_lang]).translate(text)
            st.success("✅ Translated Text:")
            st.write(f"**{translated_text}**")
        except Exception as e:
            st.error(f"❌ Translation failed: {e}")
    else:
        st.warning("⚠️ Please enter some text before translating.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Deep Translator")

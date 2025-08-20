import streamlit as st
from ML_AGENT.UI.config import Config

class SidebarUI:
    def __init__(self):
        self.user_controls = {}
        self.config = Config()

    def Load_UI(self):
        st.set_page_config(page_title="🤖 " + self.config.get_page_title(), layout="wide")
        st.header("🤖 " + self.config.get_page_title())
        with st.sidebar:
            llm_options = self.config.get_llms()
            self.user_controls["selected_llm"] = st.selectbox("Select LLM", llm_options)
            self.user_controls["llm_type"] = self.user_controls["selected_llm"]

            if self.user_controls["selected_llm"] == 'Google Gemini':
                model_options = self.config.get_gemini_llm()
                self.user_controls["select_gemini_model"] = st.selectbox("Select Model", model_options)
                self.user_controls["GOOGLE_API_KEY"] = st.session_state["GOOGLE_API_KEY"] = st.text_input("API Key", type="password", key="gemini_api_key")

                if not self.user_controls["GOOGLE_API_KEY"]:
                    st.warning("⚠️ Please enter your Google Gemini API key to proceed. Don't have? refer : https://aistudio.google.com/")

            mode = "Upload File"
            self.user_controls["mode"] = mode
            if mode == "Upload File":
                uploaded_files = st.sidebar.file_uploader(
                    "Upload your dataset",
                    type=["csv", "xlsx", "json"],
                    accept_multiple_files=True
                )
                self.user_controls["file"] = uploaded_files

        return self.user_controls

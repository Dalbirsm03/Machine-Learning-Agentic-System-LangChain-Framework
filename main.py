import streamlit as st
from langsmith import Client
import pandas as pd
from ML_AGENT.UI.Display_Results import DisplayResultStreamlit
from ML_AGENT.GRAPH.ML_Graph import Graph_Builder
from ML_AGENT.LLM.Gemini import GeminiLLM
from ML_AGENT.UI.Load_UI import SidebarUI
import os

def main():
    with st.sidebar:
        st.sidebar.title("🛠️ Configuration")
        langsmith_api_key = st.text_input("LangSmith API Key (Optional)", type="password")
        if langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
            os.environ["LANGSMITH_TRACING_V2"] = "true"
            os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
            os.environ["LANGSMITH_PROJECT"] = "ml_agent"
            try:
                Client()
                st.success("✅ LangSmith connected!")
            except Exception as e:
                st.error(f"❌ LangSmith Error: {str(e)}")
    
    sidebar = SidebarUI()
    user_controls = sidebar.Load_UI()
    st.markdown("Upload your dataset and ask ML-related questions!")

    raw_data = None
    if user_controls.get("file"):
        raw_data = [pd.read_csv(user_controls["file"][0])]
        st.success("✅ Dataset uploaded successfully!")
        st.dataframe(raw_data[0].head())

    user_message = st.chat_input("Ask me something about this dataset...")

    if user_message and raw_data is not None:
        try:
            if user_controls["llm_type"] == "Google Gemini":
                google_api_key = user_controls.get("GOOGLE_API_KEY")
                if not google_api_key:
                    st.error("Please enter your Google Gemini API Key.")
                    return
                os.environ["GOOGLE_API_KEY"] = google_api_key
                llm_object = GeminiLLM(user_contols_input=user_controls)
                llm = llm_object.get_llm_model()
                ml_graph = Graph_Builder(llm=llm)
                ml_graph = ml_graph.setup_graph()
                displayer = DisplayResultStreamlit(graph=ml_graph, user_message=user_message, raw_data=raw_data)
                displayer.display_result_on_ui()
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
    elif user_message and raw_data is None:
        st.warning("⚠️ Please upload a dataset first!")

if __name__ == "__main__":
    main()

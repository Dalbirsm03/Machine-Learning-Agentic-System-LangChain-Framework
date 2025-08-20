import os
from typing import List, Any
import streamlit as st
import pandas as pd
from ML_AGENT.STATE.ML_state import State
import logging


class DisplayResultStreamlit:
    def __init__(self, graph: Any, user_message: str, raw_data: List[pd.DataFrame]):
        self.graph = graph
        self.user_message = user_message
        self.raw_data = raw_data

    def display_result_on_ui(self):
        state = {
            "question": self.user_message,
            "raw_data": self.raw_data
        }
        with st.chat_message("user"):
            st.write(self.user_message)

        final_result = None
        for step in self.graph.stream(state, stream_mode="values"):
            if "final_result" in step:
                result = step["final_result"]
                if hasattr(result, "content"):
                    result = result.content
                final_result = result

        with st.chat_message("assistant"):
            st.markdown("🧠 **Final Analysis:**")
            if final_result:
                st.write(final_result)
            else:
                st.write("⚠️ No analysis result was generated.")


import os
from typing import List, Any
import streamlit as st
import pandas as pd
import io
import base64
from PIL import Image

class DisplayResultStreamlit:

    def __init__(self, graph: Any, user_message: str, raw_data: List[pd.DataFrame]):
        self.graph = graph
        self.user_message = user_message
        self.raw_data = raw_data
        
    def infer_task_type(self, final_result: dict) -> str:
        if not final_result:
            return "unknown"
        if "final_output" in final_result and "visual_images" in final_result:
            return "clustering"
        elif "final_result" in final_result:
            return "supervised"
        elif "final_output" in final_result:
            final_output_str = str(final_result["final_output"]).lower()
            if "cluster" in final_output_str:
                return "clustering"
            else:
                return "supervised"
        return "unknown"
    
    def display_result_on_ui(self):
        state = {
            "question": self.user_message,
            "raw_data": self.raw_data
        }

        with st.chat_message("user"):
            st.write(self.user_message)

        all_results = {}
        try:
            for step in self.graph.stream(state, stream_mode="values"):
                all_results.update(step)
        except Exception as e:
            st.error(f"Error while streaming graph results: {e}")
            all_results = {}

        with st.chat_message("assistant"):
            st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size:22px; font-weight:600;">🎯 Output</span>
        </div>
        """,
        unsafe_allow_html=True
    )
            if not all_results:
                st.write("⚠️ No analysis result was generated.")
                return

            task_type = self.infer_task_type(all_results)
            
            if task_type == "clustering":
                if "final_output" in all_results:
                    output = all_results["final_output"]
                    if hasattr(output, "content"):
                        st.write(output.content)
                    else:
                        st.write(str(output))
            if "visual_images" in all_results and all_results["visual_images"]:
                st.write("**Cluster Visualizations:**")
                for i, img in enumerate(all_results["visual_images"]):
                    # Convert PIL Image to base64
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    buf.seek(0)
                    img_bytes = buf.read()
                    img_base64 = base64.b64encode(img_bytes).decode()
                    
                    # Display centered image
                    st.markdown(
                        f"<div style='text-align: center;'>"
                        f"<img src='data:image/png;base64,{img_base64}' width='900'/>"
                        f"<p>Cluster Visualization {i+1}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            elif task_type == "supervised":
                if "final_result" in all_results:
                    output = all_results["final_result"]
                    if hasattr(output, "content"):
                        st.write(output.content)
                    else:
                        st.write(str(output))

            else:
                st.write("**Raw Results:**")
                for key, value in all_results.items():
                    if key not in ["question", "raw_data", "cleaned_data", "df"]:
                        st.write(f"**{key}:** {str(value)[:500]}...")

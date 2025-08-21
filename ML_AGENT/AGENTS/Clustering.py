from langchain_core.prompts import PromptTemplate
from ML_AGENT.STATE.ML_state import State
from ML_AGENT.logger import logger
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import io
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser , BaseOutputParser
import seaborn as sns
from typing import List, Dict, Any
import re

class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str):
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()


class Clustering_Node:
    def __init__(self, llm):
        self.llm = llm
        self.logger = logger

    def _clean_code(self, code: str) -> str:
        lines = code.strip().split('\n')
        clean_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if not in_code_block or line.strip() == '':
                continue
            clean_lines.append(line)
        
        return '\n'.join(clean_lines) if clean_lines else code.strip()

    def data_scaling(self, state: State):
        df = state['cleaned_data']        
        query = state['question'] 

        prompt = PromptTemplate(
            template="""
    You are an expert at analyzing user queries.
    The user asked: {question}

    Here are the available columns: {columns}

    Return ONLY comma-separated column names that should be used for clustering.
    Do not use quotes, brackets, or code blocks.
    Example: alcohol,density
    """,
            input_variables=["question", "columns"]
        )

        chain = prompt | self.llm | CommaSeparatedListOutputParser()
        response = chain.invoke({
            "question": query,
            "columns": list(df.columns)
        })
        selected_features = response

        if not isinstance(selected_features, list):
            self.logger.error(f"LLM did not return a list: {selected_features}")
            raise ValueError("LLM did not return a list")

        X = df[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.logger.info(f"Selected features: {selected_features}")
        return {
            "features": selected_features,
            "X_scaled": X_scaled
        }

    def find_best_k(self, state: State):
        X_scaled = state["X_scaled"]
        k_min = 2  
        k_max = 10
        best_k = None
        best_score = -1.0

        for k in range(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)

            self.logger.info(f"K={k}, silhouette score={score:.4f}")

            if score > best_score:
                best_k, best_score = k, score

        return {"best_k": best_k, "best_score": best_score}
    
    def run_clustering(self, state: State):
        X_scaled = state["X_scaled"]
        best_k = state["best_k"]
        df = state["cleaned_data"].copy()

        kmeans = KMeans(n_clusters=best_k, init="k-means++", random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        df["cluster"] = labels
        state["df"] = df  
        print(df['cluster'].value_counts())
        self.logger.info(f"Final clustering done with k={best_k}, labels attached to df")

        return {"df": df}

    def cluster_visuals(self, state: State):
        df = state["df"]
        clusters = df['cluster'].value_counts()
        columns = df.columns
        query = state["question"]
        prompt = PromptTemplate(
    template="""
You are an expert Python visualization developer specializing in clustering plots. 

Based on the user query {query} and the columns in the dataframe {columns}, selected features {selected_features} and no of cluster{clusters}, generate a Python function generate_visualizations(df) that:
1. Imports all necessary libraries inside the function.
2. Generates visualizations for multiple features:
   - If 2 numeric columns: scatter plot with clusters colored differently.
   - If more than 2 numeric columns: use simple pairplots.
3. Adds descriptive titles, axis(shoudl be little bigger) labels, and legends(should be in a proper position).
4. Calls plt.show() after each plot.
5. Makes plots readable (resize images, dpi, etc.) for production use
6. Generate only one viusalization and do not genearte elbow method visual.

Rules:
- Do not use print(), return(), placeholders, markdown, or hardcoded column names.
- Adapt dynamically to the dataframe's columns and number of numeric features.
- Make the code fully executable and production-ready.
""",
    input_variables=["query", "columns","clusters","selected_features"]
)
        try:
            chain = prompt | self.llm | StrOutputParser()
            visual_code = chain.invoke({"query": query, "columns": columns,"clusters":clusters,"selected_features": state['features'],})
            visual_code = self._clean_code(visual_code)
            self.logger.info("LLM generated visualization code successfully.")
        except Exception as e:
            self.logger.error(f"Error generating code from LLM: {e}")
            return {"visual_code": None, "visual_images": []}

        matplotlib.use("Agg")
        images: List[Image.Image] = []
        local_vars = {"df": df.copy(), "plt": plt, "sns": sns}
        original_show = plt.show

        def save_to_image():
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
            buf.seek(0)
            img = Image.open(buf)
            images.append(img.copy())
            buf.close()
            plt.close()

        try:
            exec(visual_code, {}, local_vars)
            generate_func = next((val for val in local_vars.values() if callable(val)), None)
            if generate_func is None:
                raise ValueError("No function found in generated code.")

            plt.show = save_to_image
            generate_func(df)

        except Exception as e:
            self.logger.error(f"Error executing visualization code: {e}")
        finally:
            plt.show = original_show

        return {"visual_images": images}
    
    def cluster_profile_llm(self, state: State):
        df = state["df"]
        query = state["question"]

        prompt = PromptTemplate(
            template="""
            You are an expert Python data analyst.

            The dataframe has the following columns: {features}
            The user asked: {user_query}

            Generate Python code that creates a cluster profile:
            - Group by the 'cluster' column
            - Calculate mean, std, count for numeric columns
            - For categorical columns, calculate counts and percentages
            - Use only pandas
            - Assign the final result to a variable named 'profile' (do not print or return)
            - The code should be ready to execute directly on a dataframe named 'df'

            Do not add explanations or placeholders.
            """,
            input_variables=["features", "user_query"]
        )

        chain = prompt | self.llm | PythonOutputParser()
        code = chain.invoke({
            "features": state["features"],
            "user_query": query
        })

        local_vars = {"df": df.copy(), "pd": pd, "np": np}
        exec(code, {}, local_vars)

        profile = local_vars.get("profile", None)

        prompt_summary = PromptTemplate(
        template = """
        You are an expert data analyst. 
        Based on the user query: {question} and the cluster profile: {profile}, 
        generate a **short, concise summary in bullet points only**. 
        Rules:
        .Each cluster should have max 3 to 4  lines points
        - Focus only on key insights.
        - Do not write paragraphs or explanations.
        - Use simple, clear language suitable for quick reading.
        """,
            input_variables=['question', 'profile']
        )
        chain = prompt_summary | self.llm
        response = chain.invoke({"question": query, "profile": profile})
        
        self.logger.info("Profile Generated")
        return {"final_output": response}

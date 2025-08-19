from ML_AGENT.logger import logger
from ML_AGENT.STATE.ML_state import State
from langchain_core.prompts import PromptTemplate
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import pandas as pd


class Query_Analyzer_Node:
    def __init__(self, llm):
        self.llm = llm
        self.logger = logger

    def analyze_query_and_select_target(self, state: State):
        query = state['question']
        problem_prompt = PromptTemplate(template="""
        You are an intelligent ML assistant. Analyze the user query:{query}
        Determine the type of ML problem the user is asking for: classification, regression, or clustering.
        Return only one of these three as a string.""",
            input_variables=['query']
        )
        problem_chain = problem_prompt | self.llm
        response = problem_chain.invoke({'query': query}).strip().lower()
        return {"ml_problem_type":response}
    
    def target_column(self, state: State):
        query = state['question']
        df = state['cleaned_data']
        columns = list(df.columns)

        target_prompt = PromptTemplate(
            template="""
            You are an ML assistant. The user query is:{query}
            The dataset has the following columns:{columns}
            Suggest the most likely target column for the ML problem. Return the column name as a string.
            """,
            input_variables=['query', 'columns']
        )
        target_chain = target_prompt | self.llm
        suggested_column = target_chain.invoke({'query': query, 'columns': columns}).strip()

        st.write(f"Suggested target column: **{suggested_column}**")
        user_choice = st.radio("Do you want to use this column as target?", ["Yes", "No"])
        if user_choice == "Yes":
            target_col = suggested_column
        else:
            target_col = st.selectbox("Select the target column:", options=columns)

        target_series = df[target_col]
        if target_series.dtype == object:
            le_target = LabelEncoder()
            target_encoded = le_target.fit_transform(target_series)
            state['target_label_encoder'] = le_target
        else:
            target_encoded = target_series

        state['target_encoded'] = target_encoded
        state['target_series'] = target_series

        return {
            "target_column": target_col,
            "target_series": target_series,
            "target_encoded": target_encoded
        }

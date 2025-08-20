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

    def analyze_query(self, state: State):
        self.logger.info("Starting query analysis")
        query = state['question']

        problem_prompt = PromptTemplate(
            template="""
            You are an intelligent ML assistant. Analyze the user query: {query}
            Determine the type of ML problem the user is asking for: classification, regression, or clustering.
            Return only one of these three as a string.""",
            input_variables=['query']
        )

        problem_chain = problem_prompt | self.llm
        resp = problem_chain.invoke({'query': query})
        response = resp.content.strip().lower()

        self.logger.info(f"Identified ML problem type: {response}")
        return {"ml_problem_type": response}


    def target_column(self, state: dict):
        self.logger.info("Determining target column")

        query = state['question']
        df = state['cleaned_data']
        columns = list(df.columns)
        if 'suggested_target' not in state:
            target_prompt = PromptTemplate(
                template="""
                You are an ML assistant. The user query is: {query}
                The dataset has the following columns: {columns}
                Suggest the most likely target column for the ML problem. Return the column name as a string.
                """,
                input_variables=['query', 'columns']
            )
            target_chain = target_prompt | self.llm
            resp = target_chain.invoke({'query': query, 'columns': columns})
            suggested_column = resp.content.strip()
            state['suggested_target'] = suggested_column
        else:
            suggested_column = state['suggested_target']
        if suggested_column not in columns:
            self.logger.warning(f"Suggested column '{suggested_column}' not found in dataset. Defaulting to first column.")
            suggested_column = columns[0]

        target_col = suggested_column
        self.logger.info(f"Chosen target column: {target_col}")

        state['target_column'] = target_col
        target_series = df[target_col]

        if target_series.dtype == object:
            le_target = LabelEncoder()
            target_encoded = le_target.fit_transform(target_series)
            state['target_label_encoder'] = le_target
            state['target_label_mapping'] = {
                str(cls): int(idx) for cls, idx in zip(le_target.classes_, le_target.transform(le_target.classes_))
            }


            self.logger.info(f"Applied LabelEncoder to target column. Mapping: {state['target_label_mapping']}")
        else:
            target_encoded = target_series

        return {
            "target_column": target_col,
            "target_series": target_series,
            "target_encoded": target_encoded
        }

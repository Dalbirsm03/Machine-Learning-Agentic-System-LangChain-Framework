import streamlit as st
import pandas as pd
from sklearn.feature_selection import f_classif, f_regression
from ML_AGENT.STATE.ML_state import State
from ML_AGENT.logger import logger

class Feature_Selection_Node:

    def __init__(self, llm):
        self.logger = logger
        self.llm = llm

    def select_features_streamlit(self, state: State):
        score_threshold = 0.01
        X = state['extracted_features']
        y = state['cleaned_data'][state['target_column']]
        task_type = state['ml_problem_type']

        if task_type not in ['classification', 'regression']:
            raise ValueError("ML task must be 'classification' or 'regression'.")

        if task_type == 'classification':
            scores, _ = f_classif(X, y)
        else:
            scores, _ = f_regression(X, y)

        feature_scores = pd.DataFrame({'feature': X.columns, 'score': scores})

        # Auto-select features
        auto_selected = feature_scores[feature_scores['score'] >= score_threshold]['feature'].tolist()
        if len(auto_selected) == 0:
            auto_selected = [feature_scores.sort_values('score', ascending=False)['feature'].iloc[0]]

        # Streamlit UI for user refinement
        st.write("### Feature Selection")
        st.write("Automatically selected features based on score threshold:")
        st.write(auto_selected)
        user_selected = st.multiselect(
            "Add or remove features as needed:",
            options=list(X.columns),
            default=auto_selected
        )

        final_features = list(set(user_selected))
        X_selected = X[final_features]

        self.logger.info(f"Final selected features: {final_features}")

        return {
            "X_selected": X_selected,             
            "selected_features": final_features   
        }


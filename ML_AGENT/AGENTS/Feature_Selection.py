import pandas as pd
from sklearn.feature_selection import f_classif, f_regression
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ML_AGENT.STATE.ML_state import State
from ML_AGENT.logger import logger

class Feature_Selection_Node:
    def __init__(self, llm):
        self.llm = llm
        self.logger = logger

    def select_features(self, state: State):
        X = state['extracted_features']
        y = state['target_encoded'] 
        task_type = state['ml_problem_type']
        user_query = state["question"]
        target_column = state["target_column"]
        all_columns = list(state["cleaned_data"].columns)
        available_extracted_features = list(X.columns) 


        # Ensure X contains only numeric columns
        X = X.fillna(0)  # Handle any NaN values
        X = X.astype(float)  # Convert all columns to numeric type
        
        # Ensure y is numeric for classification
        if task_type == "classification" and y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        if task_type == "classification":
            scores, _ = f_classif(X, y)
        else:
            scores, _ = f_regression(X, y)

        feature_scores = pd.DataFrame({"feature": X.columns, "score": scores})
        auto_selected = feature_scores[feature_scores["score"] >= 0.01]["feature"].tolist()
        if not auto_selected:
            auto_selected = [feature_scores.sort_values("score", ascending=False)["feature"].iloc[0]]
        prompt = PromptTemplate(
            template="""
            You are an ML expert.
            - User query: {user_query}
            - Target column: {target_column}
            - All available columns: {available_features}
            - Auto-selected features from statistical tests: {auto_selected}

            Task:
        Analyze the user query and extract the relevant features.

            Return strictly a JSON list of feature names.
            """,
            input_variables=["user_query", "target_column", "available_features", "auto_selected"]
        )

        chain = prompt | self.llm | JsonOutputParser()
        llm_suggested = chain.invoke({
            "user_query": user_query,
            "target_column": target_column,
            "available_features": available_extracted_features, # Pass the actual feature names
            "auto_selected": auto_selected
        })

        # Filter LLM suggested features to ensure they actually exist in X
        llm_suggested_filtered = [f for f in llm_suggested if f in X.columns]

        final_features = list(set(auto_selected) | set(llm_suggested_filtered))
        
        # Final fallback in case LLM and auto-selection both failed to yield valid features
        if not final_features and available_extracted_features:
            self.logger.warning("No features selected, falling back to all available extracted features.")
            final_features = available_extracted_features
        elif not final_features: # If even after fallback there are no features, raise error or return empty
            raise ValueError("No valid features could be selected for the model.")

        X_selected = X[final_features]

        self.logger.info(f"Final features after LLM enhancement: {final_features}")
        return {
            "X_selected": X_selected,
            "selected_features": final_features
        }
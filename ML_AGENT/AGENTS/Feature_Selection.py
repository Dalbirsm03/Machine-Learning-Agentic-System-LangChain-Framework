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
        y = state['cleaned_data'][state['target_column']]
        task_type = state['ml_problem_type']
        user_query = state["question"]
        target_column = state["target_column"]
        all_columns = list(state["cleaned_data"].columns)

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
            - All available columns: {all_columns}
            - Auto-selected features from statistical tests: {auto_selected}

            Task:
        Analyze the user query and extract the relevant features.

            Return strictly a JSON list of feature names.
            """,
            input_variables=["user_query", "target_column", "all_columns", "auto_selected"]
        )

        chain = prompt | self.llm | JsonOutputParser()
        llm_suggested = chain.invoke({
            "user_query": user_query,
            "target_column": target_column,
            "all_columns": all_columns,
            "auto_selected": auto_selected
        })

        final_features = list(set(auto_selected) | set(llm_suggested))
        X_selected = X[final_features]

        self.logger.info(f"Final features after LLM enhancement: {final_features}")
        return {
            "X_selected": X_selected,
            "selected_features": final_features
        }

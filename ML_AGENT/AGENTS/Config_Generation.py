from ML_AGENT.STATE.ML_state import State
from ML_AGENT.logger import logger
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


class Config_Generation_Node:
    def __init__(self,llm):
        self.llm = llm
        self.logger = logger

    def generate_config(self,state:State):
        prompt =PromptTemplate(template="""
        You are a senior ML engineer. Generate a JSON configuration for an ML pipeline based on the following inputs:
        User Query: {user_query}
        ML Task Type: {ml_type}
        Target Column: {target_column}
        Selected Features: {selected_features}

        Constraints:
        1. For classification, only use models: ["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "XGBClassifier", "SVM"].
        2. For regression, only use models: ["LinearRegression", "RandomForestRegressor", "XGBRegressor", "GradientBoostingRegressor"].
        3. Provide default hyperparameters for the chosen model.
        4. Include evaluation metrics:
        - Classification: ["accuracy", "precision", "recall", "f1-score"]
        - Regression: ["MAE", "MSE", "RMSE", "R2"]
        5. Output only JSON, with the following keys: 
        - "user_question","ml_type", "target_column", "selected_features", "model_type", "hyperparameters", "evaluation_metrics"

        Generate the JSON configuration now.
        """,
            input_variables=["user_query", "ml_type", "target_column", "selected_features"]
        )
        chain = prompt | self.llm | JsonOutputParser()
        response = chain.invoke({"user_query" : state["question"],
                                 "ml_type" : state["ml_problem_type"],
                                 "target_column" : state["target_column"],
                                 "selected_features":state["X_selected"]})
        self.logger.info("Generated ML config")
        return{"ml_config":response}
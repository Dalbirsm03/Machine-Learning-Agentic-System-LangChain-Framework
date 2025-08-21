from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
import pandas as pd
from ML_AGENT.STATE.ML_state import State
from ML_AGENT.logger import logger
import numpy as np
import re

class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str):
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

class Model_Training_Node:
    def __init__(self, llm):
        self.llm = llm
        self.logger = logger

    def model_fitting(self, state: State):
        try:
            ml_config = state["ml_config"]
            user_query = state['question']
            features = state["X_selected"]
            target = state["target_encoded"]

            self.logger.info("Starting model fitting")
            train_prompt = PromptTemplate(
                template="""
            You are a strict ML code generator.
            You will be given:

            1. User query: {user_query}
            2. ML configuration JSON: {ml_config}

            Instructions:
            - Generate Python code that:
              - Creates the model using the given `model_type` and `hyperparameters`.
              - Fits the model on the provided `X` and `y`.
            - Assume `X` and `y` are already defined in memory.
            - Store the trained model in a variable called `model`.
            - Output ONLY executable Python code, wrapped inside ```python ... ```.

            Rules:
            - Do not redefine X or y.
            - Import only what is required.
            - Do not include explanations, text, or comments.
                """,
                input_variables=["user_query", "ml_config"]
            )

            chain = train_prompt | self.llm | PythonOutputParser()
            response = chain.invoke({"user_query": user_query, "ml_config": ml_config})

            local_vars = {"X": features, "y": target}
            exec(response, {}, local_vars)
            model = local_vars["model"]

            state["trained_model"] = model
            self.logger.debug(f"Model training completed successfully{model}")
            return {"trained_model": model}

        except Exception as e:
            self.logger.error(f"Error during model fitting: {e}")
            raise
    
    def compute_metrics(self, state: State):
        try:
            X = state["X_selected"]
            y = state["target_encoded"]
            model = state["trained_model"]
            task_type = state["ml_problem_type"]

            self.logger.info("Computing model metrics")
            self.logger.debug(f"Task type: {task_type}")
            self.logger.debug(f"Evaluation data shape: {X.shape}, {y.shape}")

            y_pred = model.predict(X)

            if task_type == "classification":
                metrics = {
                    "accuracy": accuracy_score(y, y_pred),
                    "precision": precision_score(y, y_pred, average="weighted"),
                    "recall": recall_score(y, y_pred, average="weighted"),
                    "f1-score": f1_score(y, y_pred, average="weighted"),
                }
            elif task_type == "regression":
                metrics = {
                    "MAE": mean_absolute_error(y, y_pred),
                    "MSE": mean_squared_error(y, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
                    "R2": r2_score(y, y_pred),
                }
            else:
                raise ValueError("Unsupported ML task type")

            state["metrics"] = metrics
            self.logger.info(f"Metrics computed successfully: {metrics}")
            return {"metrics": metrics}

        except Exception as e:
            self.logger.error(f"Error while computing metrics: {e}")
            raise

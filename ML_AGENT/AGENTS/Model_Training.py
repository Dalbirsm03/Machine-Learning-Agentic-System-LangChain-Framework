from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
from ML_AGENT.STATE.ML_state import State
from ML_AGENT.logger import logger
import numpy as np
import re

class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str):
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

class Model_Training:
    def __init__(self, llm):
        self.llm = llm
        self.logger = logger

    def model_fitting(self, state: State):
        ml_config = state["ml_config"]
        user_query = state['question']
        features = state["X_selected"]
        target = state["target_encoded"]

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
        response = chain.invoke({
            "user_query": user_query,
            "ml_config": ml_config
        })

        local_vars = {"X": features, "y": target}
        exec(response, {}, local_vars)
        model = local_vars["model"]

        state["trained_model"] = model
        return {"trained_model": model}
    
    def compute_metrics(self, state: State):
        X = state["X_selected"]
        y = state["target_encoded"]
        model = state["trained_model"]
        task_type = state["ml_problem_type"]

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
        return {"metrics": metrics}


    def predict_from_query(self, state: State):
        user_query = state["question"]
        model = state["trained_model"]
        features = state["selected_features"]
        task_type = state["ml_problem_type"]

        predict_prompt = PromptTemplate(
            template="""
        You are an intelligent ML assistant.
        The user query is: {user_query}
        The model expects the following input features: {features}

        Extract the values for each feature from the query.
        If a value is missing, leave it as null.
        Return output strictly as a JSON dictionary: {{"feature_name": value, ...}}
            """,
            input_variables=["user_query", "features"]
        )

        chain = predict_prompt | self.llm | JsonOutputParser()
        parsed_input = chain.invoke({"user_query": user_query, "features": features})

        input_df = pd.DataFrame([parsed_input])
        prediction = model.predict(input_df)[0]

        if task_type == "classification":
            if "target_label_encoder" in state:
                prediction_label = state["target_label_encoder"].inverse_transform([prediction])[0]
            else:
                prediction_label = prediction
            result = {
                "question": user_query,
                "inputs": parsed_input,
                "prediction_encoded": int(prediction),
                "prediction_label": prediction_label
            }
        elif task_type == "regression":
            result = {
                "question": user_query,
                "inputs": parsed_input,
                "prediction": float(prediction)
            }
        else:
            raise ValueError("Unsupported ML task type")

        state["prediction_result"] = result
        return {"prediction_result": result}

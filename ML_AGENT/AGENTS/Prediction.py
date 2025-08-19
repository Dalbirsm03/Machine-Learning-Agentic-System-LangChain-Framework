from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ML_AGENT.STATE.ML_state import State
from ML_AGENT.AGENTS.Feature_Extraction import Feature_Extraction_Node
import pandas as pd

class Prediction_Node:
    def __init__(self, llm, feature_extractor):
        self.llm = llm
        self.feature_extractor = feature_extractor

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

        raw_input_df = pd.DataFrame([parsed_input])

        extracted = self.feature_extractor.extract_features({
            "cleaned_data": raw_input_df,
            "column_types": state["column_types"],
            "target_column": state["target_column"]
        })["extracted_features"]

        input_aligned = extracted.reindex(columns=state["X_selected"].columns, fill_value=0)

        prediction = model.predict(input_aligned)[0]

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
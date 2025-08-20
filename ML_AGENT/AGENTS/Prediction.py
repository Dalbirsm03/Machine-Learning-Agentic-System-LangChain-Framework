from langchain_core.prompts import PromptTemplate
from ML_AGENT.STATE.ML_state import State
from ML_AGENT.AGENTS.Feature_Extraction import Feature_Extraction_Node
from ML_AGENT.logger import logger
import pandas as pd

class Prediction_Node:
    def __init__(self, llm):
        self.llm = llm
        self.logger = logger

    def predict_from_query(self, state: State):
        user_query = state["question"]
        model = state["trained_model"]
        task_type = state["ml_problem_type"]

        self.logger.info("Generating prediction from user query")

        # Wrap query in DataFrame
        raw_input_df = pd.DataFrame([{"message": user_query}])

        # Feature extraction (reuse encoders from training)
        feature_extractor = Feature_Extraction_Node(self.llm)
        extracted = feature_extractor.extract_features({
            "cleaned_data": raw_input_df,
            "column_types": state["column_types"],
            "target_column": state["target_column"],
            "feature_encoders": state.get("feature_encoders", {}) 
        })

        extracted_features = extracted["extracted_features"]

        # Align with training columns
        input_aligned = extracted_features.reindex(columns=state["X_selected"].columns, fill_value=0)
        self.logger.info(f"Aligned input for prediction: {input_aligned.to_dict(orient='records')}")

        # Make prediction
        prediction = model.predict(input_aligned)[0]
        self.logger.info(f"Raw prediction: {prediction}")

        # Handle classification
        if task_type == "classification":
            pred_index = int(prediction)
            prediction_label = str(state["target_label_encoder"].inverse_transform([pred_index])[0])
            label_mapping = state.get("target_label_mapping", None)
            result = {
                "question": user_query,
                "inputs": {"message": user_query},
                "prediction_encoded": int(prediction),
                "prediction_label": prediction_label,
                "label_mapping": label_mapping
            }

        # Handle regression
        elif task_type == "regression":
            result = {
                "question": user_query,
                "inputs": {"message": user_query},
                "prediction": float(prediction)
            }
            label_mapping = None

        else:
            raise ValueError("Unsupported ML task type")

        state["prediction_result"] = result
        self.logger.info(f"Final prediction result: {result}")

        # Generate explanation via LLM
        explain_prompt = PromptTemplate(
            template=(
                "The result is {prediction}. Explain this in short, simple bullet points. "
                "The accuracy is {accuracy}. Explain accuracy of the model also in short, simple bullet points. "
                "If mapping is available, use it to explain classes: {mapping}"
            ),
            input_variables=["prediction", "accuracy", "mapping"]
        )
        explain_chain = explain_prompt | self.llm
        response = explain_chain.invoke({
            "prediction": prediction if task_type == "regression" else prediction_label,
            "accuracy": state.get("metrics", {}),
            "mapping": label_mapping
        })

        self.logger.info("Generated explanation for prediction")
        return {"final_result": response}

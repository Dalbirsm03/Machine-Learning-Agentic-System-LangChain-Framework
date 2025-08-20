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

        # ✅ Wrap raw user query so TF-IDF encoder can transform it
        raw_input_df = pd.DataFrame([{"message": user_query}])
        parsed_input = {"message": user_query}  # keep original text for logging

        # Run feature extraction (reuses encoders from training)
        feature_extractor = Feature_Extraction_Node(self.llm)
        extracted = feature_extractor.extract_features({
            "cleaned_data": raw_input_df,
            "column_types": state["column_types"],
            "target_column": state["target_column"]
        })

        extracted_features = extracted["extracted_features"]
        feature_encoders = extracted.get("feature_encoders", {})

        # Update encoders in state
        if "feature_encoders" not in state:
            state["feature_encoders"] = {}
        state["feature_encoders"].update(feature_encoders)

        # Align with training columns
        input_aligned = extracted_features.reindex(columns=state["X_selected"].columns, fill_value=0)
        self.logger.info(f"Aligned input for prediction: {input_aligned.to_dict(orient='records')}")

        # Make prediction
        prediction = model.predict(input_aligned)[0]
        self.logger.info(f"Raw prediction: {prediction}")

        # Handle classification vs regression
        if task_type == "classification":
            if "target_label_encoder" in state:
                prediction_label = str(state["target_label_encoder"].inverse_transform([prediction])[0])
            else:
                prediction_label = str(prediction)

            label_mapping = state["target_label_mapping"]

            result = {
                "question": user_query,
                "inputs": parsed_input,
                "prediction_encoded": int(prediction),
                "prediction_label": prediction_label,
                "label_mapping": label_mapping
            }

        elif task_type == "regression":
            result = {
                "question": user_query,
                "inputs": parsed_input,
                "prediction": float(prediction)
            }
            label_mapping = None  # ✅ keep defined

        else:
            raise ValueError("Unsupported ML task type")

        state["prediction_result"] = result
        self.logger.info(f"Final prediction result: {result}")

        # Generate explanation
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
            "prediction": result,
            "accuracy": state["metrics"],
            "mapping": label_mapping
        })

        self.logger.info("Generated explanation for prediction")
        return {"final_result": response}

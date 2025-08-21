from langchain_core.prompts import PromptTemplate
from ML_AGENT.STATE.ML_state import State
from ML_AGENT.AGENTS.Feature_Extraction import Feature_Extraction_Node
from ML_AGENT.logger import logger
import pandas as pd
import numpy as np

class Prediction_Node:
    def __init__(self, llm):
        self.llm = llm
        self.logger = logger

    def predict_from_query(self, state: State):
        user_query = state["question"]
        model = state["trained_model"]
        task_type = state["ml_problem_type"]

        self.logger.info("Generating prediction from user query")

        # Handle different input formats
        if isinstance(user_query, dict):
            raw_input_df = pd.DataFrame([user_query])
        else:
            # For text queries, create a message column
            raw_input_df = pd.DataFrame([{"message": user_query}])

        # Feature extraction (reuse encoders from training)
        feature_extractor = Feature_Extraction_Node(self.llm)
        
        try:
            extracted = feature_extractor.extract_features({
                "cleaned_data": raw_input_df,
                "column_types": state["column_types"],
                "target_column": state["target_column"],
                "feature_encoders": state.get("feature_encoders", {}) 
            })
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            # Create dummy features matching training data
            extracted_features = pd.DataFrame(0, 
                                            index=[0], 
                                            columns=state["X_selected"].columns)
            extracted = {"extracted_features": extracted_features}

        extracted_features = extracted["extracted_features"]

        # Align with training columns properly
        training_columns = state["X_selected"].columns
        input_aligned = pd.DataFrame(0, index=extracted_features.index, columns=training_columns)
        
        # Fill in matching columns
        for col in training_columns:
            if col in extracted_features.columns:
                input_aligned[col] = extracted_features[col]
                
        self.logger.info(f"Aligned input shape: {input_aligned.shape}")

        try:
            # Make prediction
            prediction = model.predict(input_aligned)
            self.logger.info(f"Raw prediction: {prediction}")

            # FIXED: Handle prediction properly
            if hasattr(prediction, '__len__') and len(prediction) > 0:
                prediction_value = prediction[0]
            else:
                prediction_value = prediction

            # Handle classification vs regression
            if task_type == "classification":
                # FIXED: Convert to integer for classification
                pred_class = int(prediction_value)
                
                if state.get("target_label_encoder") is not None:
                    try:
                        # Use inverse_transform directly - it's more reliable
                        prediction_label = str(state["target_label_encoder"].inverse_transform([pred_class])[0])
                        self.logger.info(f"Decoded prediction: {pred_class} -> {prediction_label}")
                    except Exception as e:
                        self.logger.error(f"Label decoding failed: {e}")
                        prediction_label = str(pred_class)
                else:
                    prediction_label = str(pred_class)
                    
                result = {
                    "question": user_query,
                    "inputs": raw_input_df.iloc[0].to_dict(),
                    "prediction_encoded": pred_class,
                    "prediction_label": prediction_label,
                    "label_mapping": state.get("target_label_mapping", {})
                }

            elif task_type == "regression":
                result = {
                    "question": user_query,
                    "inputs": raw_input_df.iloc[0].to_dict(),
                    "prediction": float(prediction_value)
                }

            else:
                raise ValueError(f"Unsupported ML task type: {task_type}")

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            result = {
                "question": user_query,
                "inputs": raw_input_df.iloc[0].to_dict(),
                "prediction": "Error in prediction",
                "error": str(e)
            }

        state["prediction_result"] = result
        self.logger.info(f"Final prediction result: {result}")

        # Generate explanation
        try:
            explain_prompt = PromptTemplate(
                template=(
                    "The prediction result is {prediction}. Explain this in short, simple single bullet pointsike answering the questions to user query {query}. "
                    "The model accuracy is {accuracy}. Also explain the model accuracy in simple terms and one shortest point for each. "
                    "If class mapping is available, use it to explain: {mapping}"
                ),
                input_variables=["prediction", "accuracy", "mapping","query"]
            )
            explain_chain = explain_prompt | self.llm
            response = explain_chain.invoke({
                "prediction": result.get("prediction_label", result.get("prediction", "N/A")),
                "accuracy": state.get("metrics", "Not available"),
                "mapping": state.get("target_label_mapping", "No class mapping"),
                "query":user_query
            })
            
            self.logger.info("Generated explanation for prediction")
            return {"final_result": response}
            
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            return {"final_result": f"Prediction completed: {result}"}
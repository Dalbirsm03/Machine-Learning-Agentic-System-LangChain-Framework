from langchain_core.prompts import PromptTemplate
from ML_AGENT.STATE.ML_state import State
from ML_AGENT.logger import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

class Feature_Extraction_Node:
    def __init__(self, llm):
        self.llm = llm
        self.logger = logger

    def detect_column_types(self, state: State):
        df = state['cleaned_data']
        n_rows = len(df)
        text_threshold = max(0.05, 15 / n_rows)
        target_column = state['target_column']

        col_types = {'numeric': [], 'categorical': [], 'text': [], 'datetime': [], 'boolean': []}

        for col in df.columns:
            if col == target_column:
                continue
            dtype = df[col].dtype

            if dtype in ['int64', 'float64']:
                col_types['numeric'].append(col)
            elif dtype == 'bool':
                col_types['boolean'].append(col)
            elif 'datetime' in str(dtype):
                col_types['datetime'].append(col)
            elif dtype == 'object':
                avg_len = df[col].astype(str).map(len).mean()
                if df[col].nunique() / n_rows > text_threshold or avg_len > 20:
                    col_types['text'].append(col)
                else:
                    col_types['categorical'].append(col)
            else:
                col_types['categorical'].append(col)

        self.logger.info(f"Detected column types: {col_types}")
        return {"column_types": col_types}

    def extract_features(self, state: State):
        tfidf_max_features = 40
        target_column = state['target_column']
        df = state['cleaned_data']
        col_types = state['column_types']

        if not col_types:
            raise ValueError("Column types not found in state. Run detection first.")

        extracted_features = pd.DataFrame(index=df.index)
        state.setdefault("feature_encoders", {})

        for col in df.columns:
            if col == target_column:
                continue

            dtype_category = None
            for k, v in col_types.items():
                if col in v:
                    dtype_category = k
                    break

            # FIXED: Added proper if-elif structure and indentation
            if dtype_category == "categorical":
                # Use one-hot encoding for categorical features
                one_hot_df = pd.get_dummies(df[col], prefix=col, dummy_na=False)
                extracted_features = pd.concat([extracted_features, one_hot_df], axis=1)

            elif dtype_category == "text":
                if col not in state["feature_encoders"]:
                    tfidf = TfidfVectorizer(max_features=tfidf_max_features)
                    tfidf_matrix = tfidf.fit_transform(df[col].astype(str))
                    state["feature_encoders"][col] = tfidf
                else:
                    tfidf = state["feature_encoders"][col]
                    tfidf_matrix = tfidf.transform(df[col].astype(str))

                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
                    index=df.index
                )
                extracted_features = pd.concat([extracted_features, tfidf_df], axis=1)

            elif dtype_category == "datetime":
                extracted_features[f"{col}_year"] = df[col].dt.year
                extracted_features[f"{col}_month"] = df[col].dt.month
                extracted_features[f"{col}_day"] = df[col].dt.day
                extracted_features[f"{col}_weekday"] = df[col].dt.weekday

            elif dtype_category == "numeric":
                extracted_features[col] = df[col]

            elif dtype_category == "boolean":
                extracted_features[col] = df[col].astype(int)

        self.logger.info(f"Extracted features columns: {list(extracted_features.columns)}")
        return {"extracted_features": extracted_features}
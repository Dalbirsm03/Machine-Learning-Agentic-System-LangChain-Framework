from typing_extensions import Annotated, TypedDict, Literal, Any, List, Dict, Optional
import pandas as pd
from typing import Union
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class State(TypedDict):
    # Required fields
    question: str
    ml_problem_type: str
    
    # Target-related 
    target_column: Optional[str]
    target_encoded: Optional[Union[pd.Series, np.ndarray]]
    target_label_encoder: Optional[LabelEncoder]
    target_label_mapping: Optional[Dict[int, str]]  
    
    # Data fields
    raw_data: Optional[pd.DataFrame]
    cleaned_data: Optional[pd.DataFrame]
    
    # Processing results
    data_valid: Optional[str]
    suggestions: Optional[str]
    report: Optional[str]
    
    # Feature processing
    column_types: Optional[Dict[str, List[str]]]
    extracted_features: Optional[pd.DataFrame]
    feature_encoders: Optional[Dict[str, Union[LabelEncoder, TfidfVectorizer]]]
    
    # Feature selection 
    auto_selected_features: Optional[List[str]]  
    X_selected: Optional[pd.DataFrame]
    selected_features: Optional[List[str]]  
    
    # Model training
    ml_config: Optional[Dict[str, Any]]
    trained_model: Optional[Any]
    metrics: Optional[Dict[str, Any]]
    
    # Prediction results 
    prediction_result: Optional[Dict[str, Any]]
    final_result: Optional[str]

    # Cluster 
    features : List[str]
    X_scaled : Optional[pd.DataFrame]

    best_k :str
    best_score : str
    df : pd.DataFrame
    visual_images : Any
    final_output : str


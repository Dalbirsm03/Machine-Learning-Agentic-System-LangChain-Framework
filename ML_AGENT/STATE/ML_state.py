from typing_extensions import Annotated , TypedDict , Literal, Any,List,Dict
import pandas as pd
from typing import Union

class State(TypedDict):
    question : str
    ml_problem_type : str
    target_column : str
    target_series : List[pd.DataFrame]
    target_encoded : List[pd.DataFrame]
    target_label_encoder : str
    target_label_mapping : str
    
    raw_data : List[pd.DataFrame]
    cleaned_data : List[pd.DataFrame]   
    data_valid : str
    suggestions : str
    report : str

    column_types : str
    extracted_features : List[pd.DataFrame] 
    feature_encoders : str
    
    auto_selected_features : str
    X_selected : List[pd.DataFrame] 
    selected_features : str
    ml_config : Dict[str, Any]

    trained_model : str
    metrics : Dict[str, Any]

    prediction_result : str
    final_result : str
from typing_extensions import Annotated , TypedDict , Literal, Any,List,Dict
import pandas as pd
from typing import Union
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class State(TypedDict):
    question : str
    ml_problem_type : str

    target_column: str
    target_encoded: Union[pd.Series, np.ndarray]
    target_label_encoder: LabelEncoder
    target_label_mapping: Dict[str, int]
    
    raw_data : List[pd.DataFrame]
    cleaned_data : List[pd.DataFrame]   
    data_valid : str
    suggestions : str
    report : str

    column_types : Dict[str, List[str]]
    extracted_features : List[pd.DataFrame] 
    feature_encoders: Dict[str, Union[LabelEncoder, TfidfVectorizer]]  

    auto_selected_features : str
    X_selected : List[pd.DataFrame] 
    selected_features : str
    ml_config : Dict[str, Any]

    trained_model : any
    metrics : Dict[str, Any]

    prediction_result : str
    final_result : str
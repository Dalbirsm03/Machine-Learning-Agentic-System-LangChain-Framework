from typing_extensions import Annotated , TypedDict , Literal, Any,List
import pandas as pd
from typing import Union

class State(TypedDict):
    question : str
    ml_problem_type : str
    target_column : str
    
    raw_data : List[pd.DataFrame]
    cleaned_data : List[pd.DataFrame]   
    data_valid : str
    suggestions : str
    report : str

    column_types : str
    extracted_features : List[pd.DataFrame] 

    X_selected : List[pd.DataFrame] 
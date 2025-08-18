from ML_AGENT.STATE.ML_state import State
from ML_AGENT.logger import logger
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser,BaseOutputParser
from langgraph.types import Command
from typing_extensions import Literal , List
from pydantic import BaseModel , Field
import re
import pandas as pd
import numpy as np

class Validate(BaseModel):
    valid = Literal['yes','no'] = Field(description="If data is validated then yes else no")
    suggestions :  List[str] = Field(default=[])

class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str):
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

class Data_Cleaning_Node:

    def __init__(self, llm):
        self.llm = llm
        self.logger = logger
        self.validator = llm.with_structured_output(Validate)

    def data_cleaning(self, state: State):
        try:
            raw_df = state['raw_data']
            self.logger.info("Starting data cleaning process")

            df = raw_df.copy()
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            df.drop_duplicates(inplace=True)

            missing_threshold = 0.5
            missing_ratio = df.isnull().mean()
            cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index
            if len(cols_to_drop) > 0:
                self.logger.info(f"Dropping columns with missing ratio > {missing_threshold}: {list(cols_to_drop)}")
                df.drop(columns=cols_to_drop, inplace=True)

            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col].fillna(df[col].median(), inplace=True)
                elif df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)

            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col], errors='ignore')
                    except Exception as e:
                        self.logger.warning(f"Could not convert column {col} to datetime: {e}")

            for col in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                df[col] = np.where(df[col] < lower, lower, df[col])
                df[col] = np.where(df[col] > upper, upper, df[col])

            self.logger.info("Data cleaning process completed successfully")
            return {"cleaned_data": df}
        
        except Exception as e:
            self.logger.error(f"Error during data cleaning: {e}")
            raise
        
    def data_validation(self,state : State):
        df = state['cleaned_data']

        logger.info("Generating validation report...")
        report = {
            "shape": df.shape,
            "columns": list(df.columns),
            "null_counts": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_summary": df.describe(include=[float, int], datetime_is_numeric=True).to_dict(),
            "categorical_summary": df.describe(include=[object]).to_dict(),
        }
        prompt = PromptTemplate(template = """
        You are a strict data validation system. 
        You will be given a Cleaned Data report: {report} 

        Validation Rules:
        - Null values count must be zero in cleaned data.  
        - Duplicate rows count must be zero in cleaned data.  
        - Datatypes must remain consistent and valid.  
        - No categorical or numeric summary fields should be missing.  

        Output Rules:
        - If all rules are satisfied, return exactly:
        {"data_valid": "Yes", "suggestions": []}

        - If any rule is violated, return exactly:
        {"data_valid": "No", "suggestions": ["specific, accurate, and minimal suggestions to fix the violations"]}

        - Suggestions must be accurate, clear, and directly address only the issues found.
        - Do not include any text outside the JSON/dict.
        """
        ,input_variables=['report'])
        chain = prompt | self.validator
        response = chain.invoke({"report":report})
        return {
            "report" : report,
            "data_valid": response.data_valid,
            "suggestions": response.suggestions
        }
    

    def smart_data_cleaner(self , state : State):
        prompt = PromptTemplate(template='''
            You are an intelligent Data Cleaning Agent.

            Here is the **dataset summary** (after initial cleaning):

            {report}

            Here are the **validation suggestions / issues** identified:

            {suggestions}

            Your task:
            - Generate Python code that **fixes all the issues mentioned in the suggestions**.
            - Operate only on the dataframe named `df`.
            - Always **import pandas as pd and numpy as np** in the generated code.
            - Ensure:
                1. No missing values remain.
                2. Data types are consistent.
                3. Duplicates are removed.
                4. Outliers are handled appropriately.
                5. Any other suggestion from the validator is addressed.
            - Output **only the executable Python code**, wrapped in triple backticks ```python ... ``` for easy execution.
            - Do not redefine `df`, work on the existing dataframe.
            - Do not provide any explanations, comments, or text outside the code block.
            ''', input_variables=['report', 'suggestions'])

        chain = prompt | self.llm | PythonOutputParser()
        response = chain.invoke({'report' : state["report"],'suggestions':state["suggestions"]})
        local_vars = {'df': state['cleaned_data'].copy(), 'pd': pd, 'np': np}
        try:
            exec(response, {}, local_vars)
            cleaned_data = local_vars['df']
        except Exception as e:
            self.logger.error(f"Error executing LLM cleaning code: {e}")
            raise

        self.logger.info("Smart data cleaning executed successfully")
        return Command (update={"cleaned_data": cleaned_data})

    def validate_route(self,state :State):
        logger.info(f"[Routing Decision] Cleaned status = {state['data_valid']}")
        if state["data_valid"] == 'yes':
            return "EDA Node"
        else:
            return "Smart Data Cleaning"
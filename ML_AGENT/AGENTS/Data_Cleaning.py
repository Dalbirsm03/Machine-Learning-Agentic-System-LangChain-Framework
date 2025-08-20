from ML_AGENT.STATE.ML_state import State
from ML_AGENT.logger import logger
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langgraph.types import Command
from typing_extensions import Literal, List
from pydantic import BaseModel, Field
import re
import pandas as pd
import numpy as np

class Validating(BaseModel):
    valid: Literal['yes', 'no'] = Field(description="If data is validated then yes else no")
    suggestions: List[str] = Field(default=[])

class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str):
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        code = match.group(1).strip() if match else text.strip()
        if code.startswith("0"):
            code = code[1:]
        if not any(kw in code for kw in ["df", "pd", "np"]):
            raise ValueError(f"Invalid LLM cleaning code: {code[:100]}")
        return code

class Data_Cleaning_Node:
    def __init__(self, llm):
        self.llm = llm
        self.logger = logger
        self.validator = llm.with_structured_output(Validating)

    def data_cleaning(self, state: State):
        try:
            raw_df = state['raw_data']
            self.logger.info("Starting dynamic data cleaning process")

            # Handle list of DataFrames or single DataFrame
            if isinstance(raw_df, list):
                if len(raw_df) == 1:
                    df = raw_df[0].copy()
                else:
                    self.logger.info("Multiple files detected, concatenating into a single DataFrame")
                    df = pd.concat(raw_df, ignore_index=True)
            else:
                df = raw_df.copy()

            # Standardize column names
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

            # Remove duplicate rows
            df.drop_duplicates(inplace=True)

            # Drop columns with too many missing values (>50%)
            missing_threshold = 0.5
            missing_ratio = df.isnull().mean()
            cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index
            if len(cols_to_drop) > 0:
                self.logger.info(f"Dropping columns with missing ratio > {missing_threshold}: {list(cols_to_drop)}")
                df.drop(columns=cols_to_drop, inplace=True)

            # Identify column types dynamically
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            datetime_cols = df.select_dtypes(include=['datetime']).columns

            # Fill missing values
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
            for col in datetime_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else pd.Timestamp.now())

            # Attempt to convert remaining object columns to datetime if possible
            for col in categorical_cols:
                try:
                    df[col] = pd.to_datetime(df[col], errors='ignore', infer_datetime_format=True)
                except Exception as e:
                    self.logger.debug(f"Column {col} not converted to datetime: {e}")

            # Handle numeric outliers using IQR
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                df[col] = np.where(df[col] < lower, lower, df[col])
                df[col] = np.where(df[col] > upper, upper, df[col])

            self.logger.info("Dynamic data cleaning process completed successfully")
            return {"cleaned_data": df}

        except Exception as e:
            self.logger.error(f"Error during data cleaning: {e}")
            raise



    def data_validation(self, state: State):
        df = state['cleaned_data']
        self.logger.info("Generating validation report...")

        report = {
            "shape": df.shape,
            "columns": list(df.columns),
            "null_counts": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

        # Numeric summary (only if numeric columns exist)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            try:
                numeric_summary = df[numeric_cols].describe().to_dict()
            except Exception:
                numeric_summary = {}
        else:
            numeric_summary = {}
        report["numeric_summary"] = numeric_summary

        # Categorical summary (only if object columns exist)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            try:
                categorical_summary = df[categorical_cols].describe().to_dict()
            except Exception:
                categorical_summary = {}
        else:
            categorical_summary = {}
        report["categorical_summary"] = categorical_summary

        # Datetime summary (only if datetime columns exist)
        datetime_cols = df.select_dtypes(include=["datetime"]).columns
        if len(datetime_cols) > 0:
            try:
                datetime_summary = df[datetime_cols].describe().to_dict()
            except Exception:
                datetime_summary = {}
        else:
            datetime_summary = {}
        report["datetime_summary"] = datetime_summary

        self.logger.debug(report)

        # Prompt for LLM validation
        prompt = PromptTemplate(
            template="""
    You are a strict data validation system. 
    You will be given a cleaned dataset {data}.

    Validation Rules:
    - Null values count must be zero in cleaned data.  
    - Duplicate rows count must be zero in cleaned data.  
    - Datatypes must remain consistent and valid.  
    - Numeric, categorical, or datetime summaries are allowed to be empty if no such columns exist.

    Output Rules:
    - If all rules are satisfied, return exactly:
    {{"valid": "Yes", "suggestions": []}}

    - If any rule is violated, return exactly:
    {{"valid": "No", "suggestions": ["specific, accurate, minimal suggestions to fix the violations"]}}
    """,
            input_variables=["data"]
        )

        chain = prompt | self.validator
        response = chain.invoke({"data": report})
        return {
            "data_valid": response.valid,
            "suggestions": response.suggestions
        }


    def smart_data_cleaner(self, state: State):
        prompt = PromptTemplate(
            template='''
            You are an intelligent Data Cleaning Agent.


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
            ''',
            input_variables=['suggestions']
        )
        chain = prompt | self.llm | PythonOutputParser()
        try:
            response = chain.invoke({'suggestions': state["suggestions"]})
            code_to_run = response.strip()
            local_vars = {'df': state['cleaned_data'].copy(), 'pd': pd, 'np': np}
            self.logger.debug("Executing smart cleaner code:")
            exec(code_to_run, {}, local_vars)
            cleaned_data = local_vars['df']
        except Exception as e:
            self.logger.error(f"Error executing LLM cleaning code: {e}")
            cleaned_data = state['cleaned_data']
        self.logger.info("Smart data cleaning executed successfully")
        return Command(update={"cleaned_data": cleaned_data})

    def validate_route(self, state: State):
        logger.info(f"[Routing Decision] Cleaned status = {state['data_valid']}")
        if state["data_valid"] == 'yes':
            return "Target_Column"
        else:
            return "AI_Data_Cleaner"

from configparser import ConfigParser

class Config:

    def __init__(self,config_path = "C:/Users/Dalbir/Downloads/Machine-Learning-Agentic-System-LangChain-Framework/ML_Agent/UI/config.ini"):
        self.config = ConfigParser()
        self.config.read(config_path)
    
    def get_llms(self):
        return self.config["DEFAULT"].get("LLM_OPTIONS").split(", ")
    
    def get_gemini_llm(self):
        return self.config["DEFAULT"].get("Gemini_MODEL_OPTIONS").split(", ")
    
    def get_page_title(self):
        return self.config["DEFAULT"].get("PAGE_TITLE")
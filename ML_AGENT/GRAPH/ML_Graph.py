from langgraph.graph import START , END , StateGraph
from ML_AGENT.AGENTS.Data_Cleaning import Data_Cleaning_Node
from ML_AGENT.AGENTS.Query_Analyzer import Query_Analyzer_Node
from ML_AGENT.AGENTS.Feature_Extraction import Feature_Extraction_Node
from ML_AGENT.AGENTS.Feature_Selection import Feature_Selection_Node
from ML_AGENT.AGENTS.Config_Generation import Config_Generation_Node
from ML_AGENT.AGENTS.Model_Training import Model_Training_Node
from ML_AGENT.AGENTS.Prediction import Prediction_Node
from ML_AGENT.AGENTS.Clustering import Clustering_Node
from ML_AGENT.STATE.ML_state import State
from ML_AGENT.logger import logger

class Graph_Builder:
    def __init__(self,llm,langsmith_client=None):
        self.llm = llm
        self.logger = logger
        self.langsmith_client = langsmith_client

    def ml_graph__builder(self):
        self.graph_builder = StateGraph(State)
        cleaning_node = Data_Cleaning_Node(self.llm)
        query_analyzer = Query_Analyzer_Node(self.llm)
        feature_extraction = Feature_Extraction_Node(self.llm)
        feature_selection = Feature_Selection_Node(self.llm)
        config = Config_Generation_Node(self.llm)
        model_training = Model_Training_Node(self.llm)
        prediction = Prediction_Node(self.llm)
        cluster = Clustering_Node(self.llm)


        self.graph_builder.add_node("Data_Cleaning_Agent",cleaning_node.data_cleaning)
        self.graph_builder.add_node("Data_Validating_Agent",cleaning_node.data_validation)
        self.graph_builder.add_node("AI_Data_Cleaner_Agent",cleaning_node.smart_data_cleaner)
        self.graph_builder.add_node("Query_Analyzing_Agent",query_analyzer.analyze_query)
        self.graph_builder.add_node("Target_Column_Agent",query_analyzer.target_column)
        self.graph_builder.add_node("Schema_Inspection",feature_extraction.detect_column_types)
        self.graph_builder.add_node("Feature_Extracting_Agent",feature_extraction.extract_features)
        self.graph_builder.add_node("Feature_Selection_Agent",feature_selection.select_features)
        self.graph_builder.add_node("Config_Generation",config.generate_config)
        self.graph_builder.add_node("Model_Training_Agent",model_training.model_fitting)
        self.graph_builder.add_node("Metrics_Evaluation_Agent",model_training.compute_metrics)
        self.graph_builder.add_node("Predicting_Agent",prediction.predict_from_query)
        self.graph_builder.add_node("Data_Scaling_Agent",cluster.data_scaling)
        self.graph_builder.add_node("K_Decision_Agent",cluster.find_best_k)
        self.graph_builder.add_node("Cluster_Model_Training_Agent",cluster.run_clustering)
        self.graph_builder.add_node("Cluster_Visualization_Agent",cluster.cluster_visuals)
        self.graph_builder.add_node("Cluster_Profiling_Agent",cluster.cluster_profile_llm)



        self.graph_builder.add_edge(START,"Data_Cleaning_Agent")
        self.graph_builder.add_edge("Data_Cleaning_Agent","Data_Validating_Agent")
        self.graph_builder.add_conditional_edges("Data_Validating_Agent",cleaning_node.validate_route,{"Query_Analyzing_Agent":"Query_Analyzing_Agent","AI_Data_Cleaner_Agent":"AI_Data_Cleaner_Agent"})
        self.graph_builder.add_edge("AI_Data_Cleaner_Agent","Query_Analyzing_Agent")
        self.graph_builder.add_conditional_edges("Query_Analyzing_Agent",query_analyzer.route_by_problem_type,{"unsupervised_branch":"Data_Scaling_Agent","supervised_branch":"Target_Column_Agent"})
        self.graph_builder.add_edge("Target_Column_Agent","Schema_Inspection")
        self.graph_builder.add_edge("Schema_Inspection","Feature_Extracting_Agent")
        self.graph_builder.add_edge("Feature_Extracting_Agent","Feature_Selection_Agent")
        self.graph_builder.add_edge("Feature_Selection_Agent","Config_Generation")
        self.graph_builder.add_edge("Config_Generation","Model_Training_Agent")
        self.graph_builder.add_edge("Model_Training_Agent","Metrics_Evaluation_Agent")
        self.graph_builder.add_edge("Metrics_Evaluation_Agent","Predicting_Agent")

        self.graph_builder.add_edge("Data_Scaling_Agent","K_Decision_Agent")
        self.graph_builder.add_edge("K_Decision_Agent","Cluster_Model_Training_Agent")
        self.graph_builder.add_edge("Cluster_Model_Training_Agent","Cluster_Visualization_Agent")
        self.graph_builder.add_edge("Cluster_Visualization_Agent","Cluster_Profiling_Agent")
        self.graph_builder.add_edge("Cluster_Profiling_Agent",END)


    def setup_graph(self):
        self.ml_graph__builder()
        return self.graph_builder.compile()
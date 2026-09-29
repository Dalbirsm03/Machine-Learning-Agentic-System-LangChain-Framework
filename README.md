# 🚀 Machine-Learning-Agentic-System-LangChain-Framework 

An **LLM-powered Machine Learning Agentic System** that takes raw datasets + natural language queries and produces:  
- 🧹 **Cleaned & validated data**  
- 🎯 **Supervised ML workflows** (classification, regression)  
- 📊 **Unsupervised ML workflows** (clustering)  
- 📈 **Model training, metrics, and evaluation**  
- 🖼 **Visualizations** for insights and performance  
- 🤖 **Explainable predictions** with natural language reasoning  

Built using **LangChain, LangGraph, Google Gemini Pro, LangSmith, Streamlit, FastAPI, and scikit-learn**.  

---

## 🔑 Features  

✅ **Automated Pipeline** – Data cleaning → validation → ML workflow → prediction  
✅ **Dynamic Branching** – Chooses supervised (classification/regression) or unsupervised (clustering) automatically  
✅ **Prediction Agent** – Aligns features, runs inference, decodes labels, and explains results in plain language  
✅ **Explainability** – Uses **Google Gemini Pro** for simple, human-readable explanations of results & metrics  
✅ **Interactive UI** – Upload datasets & query results via **Streamlit**  
✅ **API Ready** – Expose predictions via **FastAPI** endpoints  
✅ **Tracing & Debugging** – Full observability with **LangSmith**  

---

## 🛠 Tech Stack  

- **LangChain + LangGraph** → Agent orchestration  
- **Google Gemini Pro** → Reasoning & explanations  
- **LangSmith** → Observability & tracing  
- **scikit-learn, pandas, numpy** → Core ML engine  
- **Streamlit** → Interactive user interface  
- **FastAPI** → REST API backend  

---

## ▶️ How to Run

1. Install Python 3.10 or newer.
2. Open a terminal in the project folder and create a virtual environment:

	```powershell
	py -m venv .venv
	.venv\Scripts\Activate.ps1
	```

3. Install the project dependencies:

	```powershell
	python -m pip install -r requirements.txt
	```

4. Start the Streamlit app:

	```powershell
	streamlit run main.py
	```

5. Open the local URL shown in the terminal (usually `http://localhost:8501`). In the sidebar, enter your Google Gemini API key and upload a CSV dataset, then ask a question in the chat box.

To stop the app, press `Ctrl+C` in the terminal.

### Run the API with Postman

The FastAPI server is a separate entry point from the Streamlit app. Set your Gemini API key in PowerShell, then start the server from the project folder:

```powershell
$env:GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"
$env:GEMINI_MODEL = "gemini-2.5-flash"
python app.py
```

`GEMINI_MODEL` is optional; if omitted, the API uses `gemini-2.5-flash`. The API runs at `http://localhost:8000`. You can also open `http://localhost:8000/docs` to view its interactive API documentation. Keep this server running while sending requests from Postman. The API key is read by the server and does not need to be sent in Postman.

1. **Upload a CSV**: Create a `POST` request to `http://localhost:8000/upload`. In **Body**, choose **form-data**. Add a field named `file`, change its type to **File**, and select your CSV.
2. **Analyze the uploaded data**: Create a `POST` request to `http://localhost:8000/analyze`. In **Body**, choose **raw** and **JSON**, then send:

	 ```json
	 {
		 "question": "Which factors most affect the target?"
	 }
	 ```

The upload is kept in memory by the API process, so send both requests to the same running server. If `GOOGLE_API_KEY` is not set, `/analyze` returns a configuration error. The Streamlit interface prompts for the Gemini key separately.

---

## 🔍 Example Query  

**User:** *"Will this SMS be classified as spam?"*  
**Agent Workflow:**  
1. Cleans & validates input  
2. Selects classification branch  
3. Trains spam detector (~98.9% accuracy)  
4. Predicts label → **SPAM**  
5. Explains decision in plain language  


## 🙌 Acknowledgements  
Thanks to the **LangChain, LangGraph, LangSmith, and Streamlit** communities for enabling this project.  

---

---

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import io
import base64
from PIL import Image
from ML_AGENT.GRAPH.ML_Graph import Graph_Builder
from ML_AGENT.LLM.Gemini import GeminiLLM
import uvicorn

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

# Global variable to store uploaded data
uploaded_data = None

def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload CSV file"""
    global uploaded_data
    
    contents = await file.read()
    uploaded_data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    
    return {"message": "File uploaded successfully"}

@app.post("/analyze")
def analyze_data(request: QueryRequest):
    """Analyze data and return results"""
    global uploaded_data
    
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    
    # Set up LLM (using your existing setup)
    user_controls = {"llm_type": "Google Gemini"}
    llm = GeminiLLM(user_contols_input=user_controls).get_llm_model()
    
    # Run ML Graph
    initial_state = {"question": request.question, "raw_data": [uploaded_data.copy()]}
    ml_graph = Graph_Builder(llm=llm).setup_graph()
    
    # Collect results
    all_results = {}
    for step in ml_graph.stream(initial_state, stream_mode="values"):
        all_results.update(step)
    
    # Process results (same logic as DisplayResultStreamlit)
    response = {"question": request.question}
    
    # Check if clustering (has both text and images)
    if "final_output" in all_results and "visual_images" in all_results:
        output = all_results["final_output"]
        response["analysis"] = output.content if hasattr(output, "content") else str(output)
        
        # Convert images to base64
        if all_results["visual_images"]:
            response["images"] = []
            for img in all_results["visual_images"]:
                response["images"].append(encode_image_to_base64(img))
                
    # Check if supervised (text only)
    elif "final_result" in all_results:
        output = all_results["final_result"]
        response["analysis"] = output.content if hasattr(output, "content") else str(output)
        
    # Fallback
    else:
        response["analysis"] = str(all_results)
    
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
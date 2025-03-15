import pandas as pd
from fastapi import FastAPI, HTTPException, Form, Request
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredCSVLoader
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

app = FastAPI()

# Define Pydantic models for request and response
class AnalyzeDocumentRequest(BaseModel):
    api_key: str
    sheet_id: str
    prompt: str

class AnalyzeDocumentResponse(BaseModel):
    meta: dict
    summary: str

@app.post("/analyze_document", response_model=AnalyzeDocumentResponse)
async def analyze_csv(
    request: Request,
    api_key: str = Form(...),
    sheet_id: str = Form(...),
    prompt: str = Form(...)
):
    try:
        # Construct the Google Sheets export URL using the sheet ID
        spreadsheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        
        # Fetch the data into a DataFrame
        try:
            df = pd.read_csv(spreadsheet_url)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch data from Google Sheets: {str(e)}")
        
        # Save the DataFrame to a temporary CSV file
        result_csv_path = "output.csv"
        df.to_csv(result_csv_path, index=False)
        
        # Configure the Google Generative AI with the provided API key
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)
        
        # Load the CSV file and process it with the LLM
        loader = UnstructuredCSVLoader(result_csv_path, mode="elements")
        docs = loader.load()

        # Create the prompt template
        template1 = f"""
        Based on the following retrieved context:
        {{text}}

        You are a skilled Data Analyst. Analyze the following cluster distribution image with a focus on providing a clear summary and actionable insights. Specifically, consider the following aspects: 
        1. Summarize the main characteristics of each cluster based on the average values and distribution of features.
        2. Identify any notable patterns or trends between clusters and explain what differentiates them.
        3. Suggest potential business strategies or actions based on these cluster characteristics (e.g., marketing strategies, product offerings, customer segmentation).
        4. Provide recommendations for how each cluster might be targeted or engaged differently to maximize value.
        5. Based on the characteristics identified for each cluster, suggest a descriptive and intuitive name for each one (e.g., 'Elite Customers', 'Solid Performers').

        {prompt}
        """

        prompt_template = PromptTemplate.from_template(template1)
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        response = stuff_chain.invoke(docs)

        # Format the LLM response
        summary = response["output_text"]
        meta_info = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns.tolist()
        }

        # Construct and return the response
        return AnalyzeDocumentResponse(
            meta=meta_info,
            summary=summary
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import ChatVertexAI
from google.cloud import aiplatform

load_dotenv()

app = FastAPI(
    title="MLE Learning Agent API",
    description="An API to interact with an MLE assistant.",
    version="1.0.0",
)

# --- Vertex AI and LangChain Configuration ---
try:
    # Project and Location Configuration
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

    if not PROJECT_ID or not LOCATION:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT_ID and GOOGLE_CLOUD_LOCATION must be set."
        )
    
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    
    llm = ChatVertexAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        convert_system_message_to_human=True,
    )
    
    system_prompt = (
        "You are an expert Machine Learning Engineering (MLE) assistant. "
        "Your purpose is to provide clear, concise, and accurate explanations of MLE concepts. "
        "Be helpful, friendly, and aim to break down complex topics for users of all skill levels. "
        "When asked a question, provide a detailed answer based on your general knowledge of machine learning."
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    chain = prompt_template | llm | StrOutputParser()

except Exception as e:
    print(f"Error during initialization: {e}")    
    chain = None

# --- API Endpoint Definition ---


class QueryRequest(BaseModel):
    """Request model for the API, expecting user input."""

    input: str


class QueryResponse(BaseModel):
    """Response model for the API, providing the agent's answer."""

    response: str


@app.post(
    "/invoke",
    response_model=QueryResponse,
    summary="Get an explanation for an MLE concept",
)
def invoke_chain(request: QueryRequest):
    """
    Receives a user query, invokes the LangChain conversational chain,
    and returns the model's response.
    """
    if not chain:
        return {"response": "Error: The language model chain is not available."}

    try:
        response_text = chain.invoke({"input": request.input})
        return {"response": response_text}
    except Exception as e:
        return {"response": f"An error occurred while processing your request: {e}"}


@app.get("/", summary="Health check endpoint")
def read_root():
    """A simple health check endpoint to confirm the service is running."""
    return {"status": "MLE Assistant API is running"}


# To run this locally for testing:
# uvicorn main:app --reload

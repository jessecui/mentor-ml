import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_vertexai import ChatVertexAI
from google.cloud import aiplatform

# Project and Location Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Initialize Language Model
llm = ChatVertexAI(
    model="gemini-1.0-pro",
    temperature=0.3,
    convert_system_message_to_human=True,
)

# Create the Assistant's Persona with a System Prompt
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

# Create the Conversational Chain
chain = prompt_template | llm | StrOutputParser()

# Example Usage
if __name__ == "__main__":
    print("Testing MLE Learning Assistant Locally...")
    try:
        response_new_concept = chain.invoke(
            {
                "input": "Can you explain the difference between batch inference and online (or real-time) inference?"
            }
        )
        print("\nAgent Response:")
        print(response_new_concept)
        print("-" * 20)

        response_deeper_concept = chain.invoke(
            {
                "input": "Tell me more about CI/CD for ML. What are some common tools or platforms used?"
            }
        )
        print("\nAgent Response:")
        print(response_deeper_concept)
        print("-" * 20)

    except Exception as e:
        print(f"An error occurred during local testing: {e}")

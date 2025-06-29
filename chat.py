import requests
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def get_service_url():
    """
    Gets the service URL from one of three sources in order of priority:
    1. A command-line argument.
    2. A SERVICE_URL variable in the .env file.
    3. A manual prompt to the user.
    """
    # 1. Check for command-line argument
    if len(sys.argv) > 1:
        print("Using service URL from command-line argument.")
        return sys.argv[1]
    
    # 2. Check for environment variable (loaded from .env)
    service_url_env = os.getenv("SERVICE_URL")
    if service_url_env:
        print("Using service URL from .env file.")
        return service_url_env
        
    # 3. Prompt the user as a fallback
    print("\nCloud Run service URL not provided via command-line or .env file.")
    print("You can find the URL by running: gcloud run services describe mle-assistant-service --platform=managed --region=your-location --format='value(status.url)'")
    print("Alternatively, you can add 'SERVICE_URL=your-url' to your .env file.")
    url = input("Please paste the service URL here to continue: ")
    return url.strip()

def chat_with_api(base_url: str):
    """
    A simple command-line interface to chat with the deployed API.
    """
    endpoint = f"{base_url}/invoke"
    print("\n--- MLE Learning Assistant CLI ---")
    print("Connected to:", base_url)
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        try:
            user_input = input("\nYour question: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat. Goodbye!")
                break

            payload = {"input": user_input}
            
            print("Assistant is thinking...")
            response = requests.post(endpoint, json=payload, timeout=90)
            response.raise_for_status()

            data = response.json()
            print("\nAgent Response:")
            print(data.get("response", "No response text found."))

        except requests.exceptions.RequestException as e:
            print(f"\nAn error occurred connecting to the API: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    url = get_service_url()
    if not url:
        print("No service URL found. Exiting.")
        sys.exit(1)
    chat_with_api(url)
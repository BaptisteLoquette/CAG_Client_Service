import os
import openai
import pandas as pd
import argparse
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


load_dotenv("../../.env")

# Retrieve OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key is missing. Please set OPENAI_API_KEY in your .env file.")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Classify customer service queries for ProperShot.")
parser.add_argument("--input", type=str, required=True, help="Path to input file containing queries (one per line)", default="Logs_users.csv")
parser.add_argument("--output", type=str, required=True, help="Path to output CSV file", default="filtered_queries.csv")

args = parser.parse_args()

input_file = args.input
output_file = args.output

# System prompt to guide the LLM
system_message = SystemMessage(
    content=(
        "You are an AI assistant that filters relevant customer service queries for ProperShot,"
        " a real estate photography platform. Label each query as 1 if it is relevant based on the context,"
        " or 0 if it is irrelevant or out of scope."
    )
)

context = """
ProperShot is a real estate photography platform. Users can edit, store, and manage their property images online.
Users can download, edit, apply home staging, and process images using AI tools like the Magic Eraser.
Billing is based on a subscription model with different plans.
Issues may arise with account access, photo processing delays, or incorrect billing information.
"""

df_input = pd.read_csv(args.input)
if "Question utilisateur" not in df_input.columns:
    raise ValueError("The input CSV file must contain a 'Question utilisateur' column.")
queries = df_input["Question utilisateur"].dropna().tolist()

system_message = SystemMessage(
    content=(
        "You are an AI assistant that filters relevant customer service queries for ProperShot,"
        " a real estate photography platform. Given the following context:\n\n"
        f"{context}\n\n"
        "Label each query as 1 if it is relevant based on the context,"
        " or 0 if it is irrelevant or out of scope."
    )
)

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# Function to classify queries
def classify_query(query):
    
    response = llm([system_message, HumanMessage(content=f"Query: {query}\nLabel it as 1 (relevant) or 0 (irrelevant)")])
    label = response.content.strip()
    return label if label in ["1", "0"] else "0"  # Default to 0 if uncertain

if __name__ == "__main__":

    # Label each query
    results = [(query, classify_query(query)) for query in queries]

    df = pd.DataFrame(results, columns=["Query", "Label"])
    df.to_csv(output_file, index=False)

    print("Filtered queries saved to filtered_queries.csv") # Save to csv

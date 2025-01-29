import traceback
import gradio as gr
import pandas as pd
from scraping.scraper import UrlScraper
from rag.cagger import CAG

def process_query(query):
    # Use the global CAG_PIPE instance to generate a response
    if CAG_PIPE:
        result = CAG_PIPE.perform_cag(query)
        answer = result[0]
        error = result[1]

        if answer:
            return answer
        else:
            return error
    else:
        return "Error: CAG pipeline not initialized."

def process_csv(csv_file):
    global CAG_PIPE  # Declare CAG_PIPE as global to modify it
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            text_content = UrlScraper(df, output_path="../data/skrr.csv").scrape_return_cleaned_content()
            print(text_content)
            required_columns = ['content']
            CAG_PIPE = CAG(text_content)
            return "CSV processed successfully."
        except Exception as e:
            #return f"Error processing CSV file: {str(e)}"
            return traceback.format_exc()
    return "No CSV file uploaded."

with gr.Blocks() as demo:
    gr.Markdown("## LLM Query Interface")
    
    with gr.Row():
        query_input = gr.Textbox(lines=2, placeholder="Enter your query here...")
        query_button = gr.Button("Submit Query")
        query_output = gr.Textbox(label="Query Response")
    
    with gr.Row():
        csv_input = gr.File(
            label="Upload CSV file (must contain 'Lien vers le knowledge' and 'url' columns)",
            file_types=[".csv"],
            type="filepath"
        )

        csv_button = gr.Button("Submit CSV")
        csv_output = gr.Textbox(label="CSV Processing Result")

    csv_button.click(process_csv, inputs=csv_input, outputs=csv_output)

    query_button.click(process_query, inputs=query_input, outputs=query_output)
# Launch the application
if __name__ == "__main__":
    demo.launch()
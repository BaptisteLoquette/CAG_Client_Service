import requests
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from .utils import find_redundant_segments, remove_redundant_segments, handle_near_duplicates
from bs4 import BeautifulSoup
import requests
import pandas as pd

class UrlScraper:
    
    def __init__(self, df:pd.DataFrame, redundancy_threshold:float=0.5, near_duplicates_threshold:float=0.8):
        df["Unnamed: 2"] = df["Unnamed: 2"].astype('str')
        self.df = df
        self.redundancy_threshold = redundancy_threshold
        self.near_duplicates_threshold = near_duplicates_threshold

    def scrape_return_cleaned_content(self) -> list:
        """
        Applies the scraping and text cleaning

        Returns:list -> Cleaned text content from url's scraping
        """

        text_content = self.scrape_urls_content()

        cleaned_text_content = self.clean_text_content(text_content)

        return cleaned_text_content

    def scrape_urls_content(self) -> list:
        """
        Scrapes urls and extract text from the html pages.

        Returns:
            - html_text_content:list -> The raw html's text content
        """
        text_content = []
        

        for _, line in self.df.iterrows(): 
            url = line['Lien vers le knowledge']
            response = requests.get(url)

            if response.status_code == 200 and line['Unnamed: 2']  == "nan": # Handle edge cases where the page isn't accessible

                ## Handle .pdf content
                if url.lower().endswith('.pdf'):
                    text_content.append(self.process_pdf_from_url(url))

                ## Handle .html content
                else:
                    html_content = response.content

                    soup = BeautifulSoup(html_content, 'html.parser') # bs parse

                    text = soup.get_text(separator="\n", strip=True) # Get page's text content

                    text_content.append(text)
            else:
                text_content.append(line['Unnamed: 2'])
        print(text_content)
        
        return text_content
    
    def clean_text_content(self, text_content) -> list:
        """
        This function is used to clean the text content by removing redundant segments, handling near duplicates, and ensuring the text is in French.

        Args:
            - text_content:list -> The list of text content to clean (remove redundant segments that are probably headers or footers)

        Returns:
            - text_content:list -> The cleaned text content
        """
        redundant_segments = find_redundant_segments(text_content)
        text_content = remove_redundant_segments(text_content, redundant_segments)
        text_content = handle_near_duplicates(text_content, self.redundancy_threshold)
        return text_content
    
    def download_pdf(self, url, save_path="temp.pdf"):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            return save_path
        else:
            raise Exception("Failed to download PDF")

    def extract_text_from_pdf(self, pdf_path):
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        text = "\n".join([doc.page_content for doc in documents])
        return text

    def download_pdf(self, url, save_path="temp.pdf"):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            return save_path
        else:
            raise Exception("Failed to download PDF")

    def extract_text_from_pdf(self, pdf_path):
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        text = "\n".join([doc.page_content for doc in documents])
        return text

    def process_pdf_from_url(self, pdf_url):
        pdf_path = self.download_pdf(pdf_url)
        extracted_text = self.extract_text_from_pdf(pdf_path)
        os.remove(pdf_path)
        return extracted_text

import os
import requests
from apikey import apikey
from bs4 import BeautifulSoup
import openai
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

# Set OpenAI API Key (use env variable or replace with actual key)
os.environ["OPENAI_API_KEY"] = apikey

# Step 1: Scrape content from URL
def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.text for p in soup.find_all('p')])
        return text
    else:
        print("Failed to retrieve content")
        return ""

# Step 2: Store content in a vector database
def store_embeddings(content):
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma.from_texts([content], embeddings)
    return vector_db

# Step 3: Query the system
def query_system(query, vector_db):
    llm = ChatOpenAI(model_name="gpt-4")
    retriever = vector_db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    response = qa_chain.invoke(query)
    return response

# Step 4: Read queries from file, process, and write responses
def process_queries(input_file, output_file, vector_db):
    try:
        with open(input_file, 'r') as f:
            queries = f.readlines()

        responses = []
        for query in queries:
            response = query_system(query.strip(), vector_db)
            responses.append(f"Q: {query.strip()}\nA: {response}\n")

        with open(output_file, 'w') as f:
            f.writelines(responses)

        print("Responses written to output file")
    except FileNotFoundError:
        print(f"The file {input_file} does not exist.")
        return

# Define URL and file paths
url = "https://en.wikipedia.org/wiki/List_of_prime_ministers_of_India"
input_file = "inputquestions.txt"
output_file = "responses.txt"

# Run the pipeline
content = scrape_website(url)
if content:
    vector_db = store_embeddings(content)
    process_queries(input_file, output_file, vector_db)
else:
    print("No content retrieved from the website.")

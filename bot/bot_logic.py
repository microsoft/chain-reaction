# Description: This file contains the logic for the LLM bot
import os
import openai
import pandas as pd
import numpy as np
import json
from typing import List, Optional
import requests
from dotenv import dotenv_values
import psycopg2
from psycopg2 import pool
from psycopg2 import Error
from psycopg2 import sql
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.prompts import PromptTemplate
from langchain.llms import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
import pdb

# Get the absolute path to the .env file in the streamlit_app subdirectory
env_name = os.path.join(os.path.dirname(__file__), "llm_pgvector.env")

# Load environment variables from the .env file
config = dotenv_values(env_name)

for key, value in config.items():
    os.environ[key] = value

# LOAD OpenAI configs
openai.api_type = config["OPENAI_API_TYPE"]
openai.api_key = config['OPENAI_API_KEY']
openai.api_base = config['OPENAI_API_BASE']
openai.api_version = config['OPENAI_API_VERSION']
print("ENV VARIABLES LOADED")

## Azure cognitive search
cogsearch_name = os.getenv("COGSEARCH_NAME") #TODO: fill in your cognitive search name
cogsearch_index_name = os.getenv("COGSEARCH_INDEX_NAME") #TODO: fill in your index name: must only contain lowercase, numbers, and dashes
cogsearch_api_key = os.getenv("COGSEARCH_API_KEY") #TODO: fill in your api key with admin key

class TextFormatter(BaseLoader):
    """Load text files."""
    def __init__(self, text: str):
        """Initialize with file path."""
        self.text = text

    def load(self) -> List[Document]:
        """Load from file path."""
        metadata = {"source": ""}
        return [Document(page_content=self.text, metadata=metadata)]

def createEmbeddings(text):
    """Create embeddings for the question"""
    response = openai.Embedding.create(input=text,engine=config['OPENAI_DEPLOYMENT_EMBEDDING'])
    embeddings = response['data'][0]['embedding']
    return embeddings

def retrieve_k_chunk(k, questionEmbedding):
    ## Retrieve top K entries
	url = f"https://{cogsearch_name}.search.windows.net/indexes/{cogsearch_index_name}/docs/search?api-version=2023-07-01-Preview"
	payload = json.dumps({
		"vector": {
			"value": questionEmbedding,
			"fields": "contentVector",
			"k": k
		}
	})
	headers = {
	'Content-Type': 'application/json',
	'api-key': cogsearch_api_key,
	}
	response = requests.request("POST", url, headers=headers, data=payload)
	output = json.loads(response.text)
	print(response.status_code)
	return output

def get_context(top_rows):
	# Use the top k ids to retrieve the actual text from the database 
	top_ids = []
	for i in range(len(top_rows['value'])):
		top_ids.append(int(top_rows['value'][i]['id']))

	# Connect to the PostgreSQL database server	
	host = config["HOST"]
	dbname = config["DBNAME"] 
	user = config["USER"] 
	password = config["PASSWORD"] 
	sslmode = config["SSLMODE"] 

	# Build a connection string from the variables
	conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)

	postgreSQL_pool = psycopg2.pool.SimpleConnectionPool(1, 20,conn_string)
	if (postgreSQL_pool):
		print("Connection pool created successfully")

	# Use getconn() to get a connection from the connection pool
	connection = postgreSQL_pool.getconn()
	cursor = connection.cursor()

	# Rollback the current transaction
	connection.rollback()

	format_ids = ', '.join(['%s'] * len(top_ids))

	sql = f"SELECT CONCAT('productid: ', productid, ' ', 'score: ', score, ' ', 'text: ', text) AS concat FROM food_reviews WHERE id IN ({format_ids})"

	# Execute the SELECT statement
	try:
		cursor.execute(sql, top_ids)    
		top_rows_result = cursor.fetchall()
		for row in top_rows_result:
			print(row)
	except (Exception, Error) as e:
		print(f"Error executing SELECT statement: {e}")

	context = ""
	for row in top_rows_result:
		context += row[0]
		context += "\n"
	return context

def llm_call(context, msg):
    loader = TextFormatter(context)
    #print("Context Retrieved: {}".format(context))

    engine = config["OPENAI_MODEL_COMPLETION"]

    llm = AzureOpenAI(deployment_name=engine, model_name=config['OPENAI_MODEL_EMBEDDING'], temperature=0)

    ### Question Prompt Template
    question_prompt_template = """Use the context below to answer the question. 
    context: {context}
    question: {question}
    If the context says "No documents were found", alert the user that there was no documents. else, answer the question only using the context and cite the PageNumber and LineNumber in reference to the answer."""
    
    QUESTION_PROMPT = PromptTemplate(
        template=question_prompt_template, input_variables=["context", "question"]
        )
    print("Sending prompt...")

    chain = load_qa_chain(llm, chain_type="stuff", prompt=QUESTION_PROMPT)
    ans = chain({"input_documents": loader.load(), "question": msg}, return_only_outputs=True)

    return ans['output_text'][2:]

def test_(msg):
	questionEmbedding = createEmbeddings(msg)
	retrieve_k = 3

	top_rows = retrieve_k_chunk(retrieve_k, questionEmbedding)
	context = get_context(top_rows)
	pdb.set_trace()
	ans = llm_call(context, msg)
	return ans

if __name__ == '__main__':
	question = "Are there any plastic spoons?"

	ans = test_(question)
	print(ans)
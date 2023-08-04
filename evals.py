from dotenv import dotenv_values
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
import time
import csv
import yaml

# Get filename of .env
with open('config.yaml', 'r') as f:
    tool_config = yaml.safe_load(f)

# specify the name of the .env file name 
env_name = tool_config['env_file_name'] # change to use your own .env file
config = dotenv_values('bot/'+env_name+'.env')

## Configure openai
openai.api_type = config["OPENAI_API_TYPE"] #"azure"
openai.api_key = config['OPENAI_API_KEY']
openai.api_base = config['OPENAI_API_BASE'] 
openai.api_version = config['OPENAI_API_VERSION'] 


def createEmbeddings(text):
    response = openai.Embedding.create(input=text , engine=config["OPENAI_DEPLOYMENT_EMBEDDING"])
    embeddings = response['data'][0]['embedding']
    return embeddings

def get_cosine_similarity(A_true, A_pred):
    # A_true: list of true answers
    # A_predicted: list of predicted answers
    
    # cosine_similarity_score: average score over all the questions
    # cosine_similarities: list of cosine similarities
    
    cosine_similarities = []
    for i in range(len(A_true)):
        emb_true = createEmbeddings(A_true[i])
        time.sleep(0.1)
        emb_pred = createEmbeddings(A_pred[i])
        cosine_similarity_val = cosine_similarity(
            np.array(emb_true).reshape(1, -1), np.array(emb_pred).reshape(1, -1)
        )[0][0]
        cosine_similarities.append(np.round(cosine_similarity_val, 4))
        
    cosine_similarity_score = sum(cosine_similarities) / len(cosine_similarities)
    
    return cosine_similarity_score, cosine_similarities

## This function is used to extract the list of questions and answers from the csv files
def returnQA(benchmark_csv_file_path, solution_csv_file_path):
    ## Extract list of questions and answers given csv files
    
    Questions = []
    A_true = []
    A_pred = []
    
    with open(benchmark_csv_file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            Questions.append(row["Question"])
            A_true.append(row["Answer"])
    
    with open(solution_csv_file_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            A_pred.append(row["Answer"])
            
    return Questions, A_true, A_pred

def returnQA_(csv):
    Questions = []
    A_true = []
    A_pred = []

    with open(csv, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            Questions.append(row["question"])
            A_true.append(row["Answer"])
            A_pred.append(row["predicted"])
    return Questions, A_true, A_pred


if __name__ == "__main__":
    A_true = ["Hi all", "my name is"]
    A_pred = ["Hello all", "my name is not"]

    get_cosine_similarity(A_true, A_pred)


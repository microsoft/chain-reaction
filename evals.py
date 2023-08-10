from dotenv import dotenv_values
from sklearn.metrics.pairwise import cosine_similarity
from langchain import PromptTemplate, LLMChain
import numpy as np
import openai
import time
import csv
import yaml
import pdb

# Get vars from config file
with open('config.yaml', 'r') as f:
    tool_config = yaml.safe_load(f)

# specify the name of the .env file name 
env_name = tool_config['env_file_name'] # change to use your own .env file
config = dotenv_values('bot/'+env_name+'.env')

## Configure openai
openai.api_type = config["OPENAI_API_TYPE"]
openai.api_key = config['OPENAI_API_KEY']
openai.api_base = config['OPENAI_API_BASE']
openai.api_version = config['OPENAI_API_VERSION'] 

def createEmbeddings(text):
    response = openai.Embedding.create(input=text, engine=config['OPENAI_DEPLOYMENT_EMBEDDING'])
    embeddings = response['data'][0]['embedding']
    return embeddings

def get_cosine_similarity(A_true, A_pred):
    # A_true: str of true answers
    # A_predicted: str of predicted answers
    
    # cosine_similarity_score: score of cosine similarity between A_true and A_pred
    
    emb_true = createEmbeddings(A_true)
    emb_pred = createEmbeddings(A_pred)
    cosine_similarity_val = cosine_similarity(
        np.array(emb_true).reshape(1, -1), np.array(emb_pred).reshape(1, -1)
    )[0][0]
    cosine_similarity_score = np.round(cosine_similarity_val, 4)
        
    return cosine_similarity_score

def AI_similarity(A_true, A_pred, Question):
    """
    Evaluate the semantic similarity between the predicted and true answers using an llm model.

    Args:
        A_true: str of true answers.
        A_pred: str of predicted answers from the language model.
        Question: str of questions corresponding to each answer.

    Returns:
        ai_score (int): Semantic similarity score between the predicted and true answers.
    """
    
    val_template = """ 
    Question: {Question}
    PredictedAnswer: {PredictedAnswer}
    TrueAnswer: {TrueAnswer}
    Evaluate the accuracy of the question answering model's predictions.
    Given a question, the true answer (TrueAnswer), and the model's prediction (PredictedAnswer), determine if the model's prediction matches the true answer.
    Return only a single score of 0 or 1 indicating whether the model's prediction is correct. The response should be an integer with no newline characters."""

    ai_score = "NA"
        
    val_prompt = val_template.format(
        Question=Question, 
        PredictedAnswer=A_pred,
        TrueAnswer = A_true
    )
    response = openai.Completion.create(
        engine=config["OPENAI_DEPLOYMENT_COMPLETION"],
        prompt=val_prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0,
    )
    ans = response['choices'][0]['text']
    ai_score = int(ans)
    return ai_score

if __name__ == "__main__":
    Question = "How do you introduce yourself?"
    A_true = "My name is"
    A_pred = "My name is not"

    print('Cosine similarity:', get_cosine_similarity(A_true, A_pred))
    print('AI similarity:', AI_similarity(A_true, A_pred, Question))
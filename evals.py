from dotenv import dotenv_values
from sklearn.metrics.pairwise import cosine_similarity
from langchain import PromptTemplate, LLMChain
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
openai.api_base = config['OPENAI_API_ENDPOINT'] 
openai.api_version = config['OPENAI_API_VERSION'] 
embeddings_deployment = config['OPENAI_EMBEDDINGS_DEPLOYMENT']
completions_deployment = config['OPENAI_COMPLETIONS_DEPLOYMENT']


def createEmbeddings(text):
    response = openai.Embedding.create(input=text , engine=embeddings_deployment)
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

def AI_similarity_v0_text_davinci(A_true, A_pred, Question):
    '''
    Input: True Answer, LLM answer, Question for which questions are asked. 
    Output: LLM evaluation of semantic similarity: float [0,1]
    '''
    val_template = """ 
    Question: {Question}
    PredictedAnswer: {PredictedAnswer}
    TrueAnswer: {TrueAnswer}
    Calculate and provide the similarity score between the predicted answer (PredictedAnswer) and the true answer (TrueAnswer) for a given question (Question).

    In cases where PredictedAnswer bears no similarity to TrueAnswer, the output should be 0.
    When PredictedAnswer and TrueAnswer are semantically identical, the output should be 1.
    For all other scenarios, the output should be a floating-point number between 0 and 1, reflecting the extent of similarity between the two answers.
    Ensure that your output is formatted correctly: it should be a floating-point number between 0 and 1, without any additional text such as 'answer=' or 'output ='. For instance, '1' and '0.7' are correct formats, while 'answer= 1' and 'output =0.7' are not.

    Return the similarity score as the sole output."""

    ai_score = "NA"
    scores = len(Question)*["NA"]
        
    try:
        if completions_deployment != "text-davinci-003":
            raise ValueError("Deployment must be tex-davinci-003")
        scores = []
        for i in range(len(Question)):
            val_prompt = val_template.format(Question = Question[i], PredictedAnswer = A_pred[i], TrueAnswer = A_true[i])
            response = openai.Completion.create(
            engine= completions_deployment,
            prompt=val_prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=1,
        )
            print(response)
            ans = response['choices'][0]['text']
            print(ans)
            scores.append(float(ans))
        ai_score = sum(scores)/len(scores)
    except ValueError as e:
        print(f"Error: {e} ")


    return ai_score, scores

def AI_similarity_v1_text_davinci(A_true, A_pred, Question):
    '''
    Input: True Answer, LLM answer, Question for which questions are asked. 
    Output: LLM evaluation of semantic similarity: Boolean Value: TRUE/FALSE
    '''
    val_template = """ 
    Question: {Question}
    PredictedAnswer: {PredictedAnswer}
    TrueAnswer: {TrueAnswer}
    Evaluate the accuracy of the question answering model's predictions.
    Given a question, the true answer (TrueAnswer), and the model's prediction(PredictedAnswer), determine if the model's prediction matches the true answer.
    Return a Boolean value indicating whether the model's prediction is correct (True/False).
    """
    ai_score = "NA"
    scores = len(Question)*["NA"]

    try:
        if completions_deployment != "text-davinci-003":
            raise ValueError("Deployment must be tex-davinci-003")
        scores = []
        for i in range(len(Question)):
            val_prompt = val_template.format(Question = Question[i], PredictedAnswer = A_pred[i], TrueAnswer = A_true[i])
            response = openai.Completion.create(
            engine= completions_deployment,
            prompt=val_prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=1,
        )
            print(response)
            ans = response['choices'][0]['text']
            print(ans)
            scores.append(float(eval(ans.strip())))
            print(float(eval(ans.strip())))
        ai_score = sum(scores)/len(scores)
    except ValueError as e:
        print(f"Error: {e} ")
    
    return ai_score, scores

if __name__ == "__main__":
    A_true = ["Hi all", "my name is"]
    A_pred = ["Hello all", "my name is not"]
    Question = ["What to say first when meeting a group of people?", "How do you introduce yourself?"]

    print('Cosine similarity:', get_cosine_similarity(A_true, A_pred))
    print('AI similarity:', AI_similarity_v1_text_davinci(A_true, A_pred, Question))

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

def get_llm_based_similarity_v1(A_true, A_pred, Question):
    '''
    Input: True Answer, LLM answer, Question for which questions are asked. 
    Output: LLM evaluation of semantic similarity
    '''
    val_template = """ 
    Question: {Question}
    PredictedAnswer: {PredictedAnswer}
    TrueAnswer: {TrueAnswer}
    Analyze similarity between the PredictedAnswer and TrueAnswer for a given Question.
    Your output is 0 if the PredictedAnswer is completely different from the correct TrueAnswer. 
    Your output is 1 if the PredictedAnswer is semantically same as the correct TrueAnswer . 
    Otherwise, your output is between 0 and 1 depending on the similarity of the answer and the correct answer.
    Double check your output format before submitting it as output should not include anything other than a floating number from 0 to 1.
    """

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
    return ai_score, scores

def get_llm_based_similarity_v2(A_true, A_pred, Question):
    '''
    Input: True Answer, LLM answer, Question for which questions are asked. 
    Output: LLM evaluation of semantic similarity
    '''
    val_template = """ 
    Question: {Question}
    PredictedAnswer: {PredictedAnswer}
    TrueAnswer: {TrueAnswer}
    Analyze similarity between the PredictedAnswer and TrueAnswer for a given Question.
    Your output is 0 if the answer is completely different from the correct answer. 
    Your output is 1 if the answer is semantically same as the correct answer. 
    Otherwise, your output is between 0 and 1 depending on the similarity of the answer and the correct answer.
    Format the output as a number between 0 and 1, including 0 and 1.
    Double check your output format before submitting it as output should not include anything other than a floating number from 0 to 1.
    """
    from langchain.llms import AzureOpenAI
    from langchain import PromptTemplate, LLMChain
    from langchain.schema import HumanMessage

    messages = [
        HumanMessage(content=text),
    ]
    print("From LLM: ")
    agent.run(messages)

    prompt_template = PromptTemplate(
        input_variables=["Question", "PredictedAnswer", "TrueAnswer"],
        template=val_template,
    )
    azure_llm = AzureOpenAI(
    deployment_name=completions_deployment,
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base,
    openai_api_version=openai.api_version,
    )

#     openai.api_type = config["OPENAI_API_TYPE"] #"azure"
# openai.api_key = config['OPENAI_API_KEY']
# openai.api_base = config['OPENAI_API_ENDPOINT'] 
# openai.api_version = config['OPENAI_API_VERSION'] 

    llm_chain = LLMChain(llm=azure_llm, prompt=prompt_template)
    scores = []
    for i in range(len(Question)):
        ans = llm_chain.predict(
            Question = Question[i],
            PredictedAnswer = A_pred[i],
            TrueAnswer = A_true[i],
        )
        print(ans)
        scores.append(float(ans))
    ai_score = sum(scores)/len(scores)
    return ai_score, scores

def get_llm_based_similarity_v3(A_true, A_pred, Question):
    '''
    Input: True Answer, LLM answer, Question for which questions are asked. 
    Output: LLM evaluation of semantic similarity
    '''
    val_template = """ 
    Question: {Question}
    PredictedAnswer: {PredictedAnswer}
    TrueAnswer: {TrueAnswer}
    Analyze similarity between the PredictedAnswer and TrueAnswer for a given Question.
    Your output is 0 if the answer is completely different from the correct answer. 
    Your output is 1 if the answer is semantically same as the correct answer. 
    Otherwise, your output is between 0 and 1 depending on the similarity of the answer and the correct answer.
    Format the output as a number between 0 and 1, including 0 and 1.
    Double check your output format before submitting it as output should not include anything other than a floating number from 0 to 1.
    """
    from langchain.llms import AzureOpenAI
    from langchain import PromptTemplate, LLMChain
    from langchain.schema import HumanMessage


    # prompt_template = PromptTemplate(
    #     input_variables=["Question", "PredictedAnswer", "TrueAnswer"],
    #     template=val_template,
    # )
    azure_llm = AzureOpenAI(
    deployment_name=completions_deployment,
    openai_api_key=openai.api_key,
    openai_api_base=openai.api_base,
    openai_api_version=openai.api_version,
    )

#     openai.api_type = config["OPENAI_API_TYPE"] #"azure"
# openai.api_key = config['OPENAI_API_KEY']
# openai.api_base = config['OPENAI_API_ENDPOINT'] 
# openai.api_version = config['OPENAI_API_VERSION'] 

    llm_chain = LLMChain(llm=azure_llm)
    scores = []
    for i in range(len(Question)):
        val_prompt = val_template.format(Question = Question[i], PredictedAnswer = A_pred[i], TrueAnswer = A_true[i])
        messages = [
            HumanMessage(content=val_prompt),
                ]
        ans = llm_chain.predict(messages
        )
        print(ans)
        scores.append(float(ans))
    ai_score = sum(scores)/len(scores)
    return ai_score, scores

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
    Question = ["What to say first when meeting a group of people?", "How do you introduce yourself?"]

    print('Cosine similarity:', get_cosine_similarity(A_true, A_pred))
    print('AI similarity:', get_llm_based_similarity_v3(A_true, A_pred, Question))
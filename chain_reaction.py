"""
Tool for evaluating LLM performance on question answering with metrics like
cosine similarity, AI similarity, logprob etc. Metrics will be logged with 
MLFlow. Make sure each function explicitly returns variables in 
tuple/single string, but no dictionary

Usage:

python chain_reaction.py
"""

__author__ = "Journey McDowell"
__copyright__ = "Copyright 2023, Microsoft Corp."

import numpy as np
import pandas as pd
import yaml
from evals import get_cosine_similarity
import importlib
import json
import time
import sys
import types
import traceback
import mlflow
import warnings
import pdb

warnings.simplefilter('ignore')
RETRIES = 2

def call_function_get_frame(func, *args, **kwargs):
      """
      Calls the function *func* with the specified arguments and keyword
      arguments and snatches its local frame before it actually executes.
      """
      frame = None
      trace = sys.gettrace()
      def snatch_locals(_frame, name, arg):
        nonlocal frame
        if frame is None and name == 'call':
          frame = _frame
          sys.settrace(trace)
        return trace
      sys.settrace(snatch_locals)
      try:
        result = func(*args, **kwargs)
      finally:
        sys.settrace(trace)
      return frame, result

def namespace_decorator(func):
    """
    Decorator that returns a module with the local variables of the function
    """
    def wrapper(*args, **kwargs):
        frame, result = call_function_get_frame(func, *args, **kwargs)
        try:
            module = types.ModuleType(func.__name__)
            module.__dict__.update(frame.f_locals)
            return module, result
        finally:
            del frame
    return wrapper

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use importlib to import the chain functions
chain_fxns = importlib.import_module('bot.'+config['bot_logic_file_name'])
chain_instruct = config['chain_instruct']

# MLFlow logging
mlflow_experiment_name = config["mlflow_experiment_name"]
mlflow_run_name = config["mlflow_run_name"]

mlflow.set_experiment(experiment_name=mlflow_experiment_name)

# Read in benchmark csv
df = pd.read_csv('bot/'+config['benchmark_csv']+'.csv')

df_result = df[['Question', 'Answer']]
df_result['generated'] = np.nan

# Scoring method(s)
scoring_methods = [key for key, val in config['scoring'].items() if val == True]

for method in scoring_methods:
    df_result[method] = np.nan

df_result['chain_error'] = np.nan
df_result['metric_error'] = np.nan

df_result['time'] = np.nan

# Initialize function vars to log and sequence 
for key in config['function_logging_vars']:
    df_result[key] = np.nan

for key, val in config['chain_instruct'].items():
    for outp in val['out']:
        df_result[outp] = np.nan

with mlflow.start_run(run_name=mlflow_run_name) as run:
    mlflow.log_params(config['constants'])
    mlflow.log_param('sequence', list(config['chain_instruct'].keys()))
    grace_stop = False
    for i, row in df.iterrows():
 
        raw_msg = row['Question']
        a_true = row['Answer']

        # Find the first input variable from the dictionary chain_instruct
        input_var = list(chain_instruct.values())[0]['in'][0]
        # Initialize hash map for inputs/outputs of each chain function
        link = {input_var: raw_msg}
        link.update(config['constants'])
        for key, val in chain_instruct.items():
            start = time.time()
            if key in chain_fxns.__dict__.keys():
                # Use wrapper to get local variables
                fxn = namespace_decorator(chain_fxns.__dict__[key])
                
                # Run chain function with args from schema
                for attempt in range(RETRIES):
                    # Rate limit
                    time.sleep(0.1)

                    try:
                        result_module, result = fxn(
                            *(link[inp] for inp in val['in'])
                        )

                        # Check if config['function_logging_vars'] keys are in result_module
                        for hyp_key in config['function_logging_vars']:
                            if hyp_key in result_module.__dict__.keys():
                                df_result[hyp_key][i] = result_module.__dict__[hyp_key]
                        
                        # Update link dictionary with outputs
                        if len(result) > 1 and len(val['out']) > 1:
                            [link.update({outp: result[i]}) for i, outp in enumerate(val['out'])]
                        else:
                            [link.update({outp: result}) for outp in val['out']]
                    
                    except Exception:
                        print("Error at chain {}".format(key))
                        print(traceback.format_exc())
                        df_result['chain_error'][i] = traceback.format_exc()
                        
                        if attempt == RETRIES - 1:
                            grace_stop = True
                    else:
                        # Successful try, no need to retry
                        break
            else:
                print("Function name {} is not available in {}.py".format(key, config['bot_logic_file_name']))
                exit()

            # On final chain, calculate metrics
            if key == list(chain_instruct.keys())[-1]:
                # Log time
                end = time.time()
                df_result['time'][i] = np.round(end - start, 4)

                df_result['generated'][i] = link[val['out'][0]]

                # Choose scoring method and utilize
                for method in scoring_methods:
                    if method == 'cosine_similarity':
                        for attempt in range(RETRIES):
                            try:
                                cosine_similarity_score, cosine_similarities = get_cosine_similarity(a_true, link[val['out'][0]])
                                time.sleep(0.1)
                                df_result[method][i] = np.round(cosine_similarity_score, 4)
                            except Exception:
                                print("Error at cosine similarity")
                                print(traceback.format_exc())
                                df_result['metric_error'][i] = traceback.format_exc()

                                if attempt == RETRIES - 1:
                                    grace_stop = True
                            else:
                                # Successful try, no need to retry
                                break

                    elif method == 'ai_similarity':
                        # TODO: Implement AI similarity
                        pass
                    else:
                        print("No valid metric specified in config.yaml")
                        exit()

                # Log each output of each chain function
                for key, val in config['chain_instruct'].items():
                    for outp in val['out']:
                        if type(link[outp]) == list or type(link[outp]) == dict:
                            # Do not log things like embeddings
                            pass
                        else:
                            df_result[outp][i] = link[outp]

            # Grace stop if too many retries - this is inner loop
            if grace_stop == True:
                break

            # End of chain instruct loop
        
        # Grace stop if too many retries - this is outer loop
        if grace_stop == True:
            print("Stopping experiment early due to error after {} attempts...".format(RETRIES))
            break

        # End row loop

    for method in scoring_methods:	
        mlflow.log_metrics({'avg_'+method: np.round(
                np.mean(df_result[method].dropna()), 4
            )
        })

    df_result['generated'] = df_result['generated'].replace(r'^\s*$', np.nan, regex=True)
    mlflow.log_metrics({'num_complete_chain': len(df_result['generated'].dropna())})
    
    df_result.to_csv('results.csv', index=False)
    mlflow.log_artifact('results.csv')

mlflow.end_run()

print("Experiment complete, please run the following command and visit localhost:5000 in browser: mlflow ui")



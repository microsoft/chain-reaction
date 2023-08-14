# Chain Reaction - Experimentation Tool for LLMs

`Chain Reaction` is a Python based LLM experimentation tool that reads instructions of function names to link in a sequence from a `config.yaml`, ingests benchmark question/answer(s) from a `.csv`, and logs/reports info/metrics about the answers from an LLM to `MLFlow`.

The principle behind the design of `Chain Reaction` is to not have to do code integration with a bot since that itself is changing when one wants to swap out functions like to retrieve or parse in a chain to evaluate its performance in the end. Metrics like cosine similarity, AI similarity, and logprob are used to score the performance of your LLM app. 

Some assumptions are made:
- chains are sequenced with Python functions and not inner calls of langchain/semantic_kernel
- input/output of each function must be string or tuple (no dictionary)
- desired variables to be logged within the scope of a function must be defined as a variable in `locals()` since we are retrieving from the stack
- desired variables to be logged within the scope of a function must not be lists or dictionaries.

![](img/chain_reaction_design.png)

## Setup

```cmd
chain-reaction                            
|---chain_reaction.py            
│---config.yaml                  
│---environment.yaml              
│---evals.py                     
├───bot                          
    │---example.env         
    │---llm_app.py                
    │---benchmark_QA.csv         
    │---prompt_templates.csv         
```

*** Note: For specific logic to allow experimenting, add `llm_app.py` separately. Some example connections with custom databases and integrations with openai are presented here: https://github.com/microsoft/AzureDataRetrievalAugmentedGenerationSamples/tree/main/Python. Please end-to-end examples are shared separately. 

1. Install miniconda and create a virtual environment
```bash
conda env update -f environment.yaml
```
2. Place relevant bot files like `.env`, `.py` into the folder named `/bot`
3. Update the `.env` file to include OpenAI embedding models
4. Install requirements for your bot
5. Update the `config.yaml` with your `llm_app_file_name` from the `/bot` folder
6. Place `.csv` of benchmark Q&A in the folder named `/bot`

| Question      | Answer |
| ----------- | ----------- |
| What is the age of the universe? | Scientists estimate that the age of the universe is around 13.8 billion years. |
| What is the composition of the universe? | The universe is primarily made up of dark energy, dark matter, and ordinary matter, with ordinary matter being the most familiar to us. |

7. Update the `config.yaml` with your `benchmark_csv` from the `/bot` folder
8. Update the `config.yaml` with the in/out variable names in your chain sequence, be sure to define the `input_var` for your LLM message variable name in case your initial function in the chain sequence takes different inputs
9. Update the `config.yaml` with any `experiment_vars` you want to set as args for functions
10. Update the `config.yaml` with variable names you want to log from function scopes under `internal_logged_vars`

Example config file below:
```yaml
llm_app_file_name: llm_app
benchmark_csv: benchmark_QA
env_file_name: llm_pgvector
mlflow_experiment_name: Default
mlflow_run_name: run 1
evaluation_metrics:
  cosine_similarity: true
  ai_similarity: false
input_var: msg
experiment_vars:
  prompt_templates_name: prompt_templates
  prompt_id_a: 2
  prompt_id_b: 2
internal_logged_vars:
  - engine
chain_instruct:
  - fxn_name: get_prompt
    in:
      - prompt_templates_name
      - prompt_id_a
    out:
      - prompt_template_a
  - fxn_name: llm_call
    in:
      - msg
      - prompt_template_a
    out:
      - result
  - fxn_name: get_prompt
    in:
      - prompt_templates_name
      - prompt_id_b
    out:
      - prompt_template_b
  - fxn_name: llm_call
    in:
      - msg
      - prompt_template_b
    out:
      - ans
```

## Usage

After updating your `config.yaml`, run the following command to loop through the benchmark Q&A csv and report out the results `MLFlow`. You can also specify a different config file by using the optional flag `--config=<insert_name>.yaml`.

```bash
python chain_reaction.py
```

> NOTE: There is some error handling with running a function call and if it fails to run, the remaining sequence will not complete. The next question will be prepped and evaluated instead. The logs will be missing information when this happens.

Once the results are complete, you will be able review your experiment alongside other ones in `MLFlow`. The cleanest view of the results are actually in a csv and a preview can be found in `MLFlow`.

```bash
mlflow ui
```
![](img/mlflow_dashboard.png)

## Adding New Metrics

In `evals.py`, simply add a new function and then import it directly in `chain_reaction.py`. We will need to add some logic to allow a user to specify the new metric in `config.yaml` based on true/false. 

## Maintainer

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

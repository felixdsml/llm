# LLM Project

## Overview

This repository is a part of my thesis work focused on the development, deployment, and evaluation of a language model. Below is a description of the folder structure and the purpose of each file and directory.


### Directory and File Descriptions

- **`convert-to-pdf/`**
  - `convert_to_pdf.ipynb`: Jupyter notebook for converting project documents or outputs into PDF format, likely used for generating final reports or figures.

- **`deployment/`**
  - `prompts.txt`: A text file containing prompts used during the deployment or evaluation of the language model.
  - `streamlit-openai-chat.py`: A Python script for deploying a Streamlit application that interacts with the OpenAI API, allowing for real-time chat with the language model.

- **`evaluation-pipeline/`**
  - `clean_and_commented_json.py`: A Python script to clean and comment JSON files, possibly used for pre-processing evaluation data.
  - `clean_and_commented_json_openai.py`: Similar to the above script but tailored for use with OpenAI-related JSON data.
  - **`evaluation/`**
    - `evaluation.ipynb`: A Jupyter notebook containing the main evaluation process for the language model.
    - `figures/`: Directory for storing figures generated during the evaluation process.
    - `latex/`: Directory potentially for LaTeX files related to the evaluation, perhaps for generating report-ready figures or tables.
    - `log_evaluations.xlsx`: An Excel file tracking various evaluation logs.
    - `model_sizes.xlsx`: An Excel file documenting the sizes of different models used during evaluation.
  - `log_evaluations.xlsx`: A duplicate or separate Excel file for tracking evaluation logs.
  - `logs/`: Directory for storing log files generated during the evaluation process.
  - `optimized_programs/`: Directory likely containing optimized versions of scripts or programs used in the evaluation.
  - **`thesis_figures/`**
    - `prompting_techniques_hierarchy.png`: An image file showing a hierarchy of prompting techniques, used in the thesis.
    - `thesis_figures.ipynb`: A Jupyter notebook dedicated to generating figures specifically for the thesis.
  - `utils_evaluate.py`: Utility script with functions and helpers for the evaluation process.
  - `utils_ollamalocal.py`: A utility script, possibly related to local deployment or testing with a specific model named "Ollama."
  - `utils_random_search.py`: A script for performing random search optimization, possibly for hyperparameter tuning during model evaluation.

- **`folder_structure.txt`**
  - A text file containing the structure of the project, likely for documentation purposes.

- **`litellm-docker/`**
  - `config_base.yaml`: A base YAML configuration file for deploying the lightweight version of the LLM.
  - `config_evaluator.yaml`: YAML configuration tailored for evaluating the lightweight LLM.
  - `docker-compose-eval.yml`: Docker Compose file for setting up an evaluation environment.
  - `docker-compose_base.yml`: Docker Compose file for setting up the base deployment environment.

- **`phoenix-docker/`**
  - `docker-compose.yml`: Docker Compose file for deploying the LLM within a Phoenix framework-based environment.

- **`unsloth_tunes/`**
  - `Alpaca_+_Llama_3_8b_full_example_sql_edit_checkpoint.ipynb`: A Jupyter notebook for working with the Alpaca and Llama models, focused on SQL-related tasks or checkpoints.
  - `Llama_3_8b_chat_template_Unsloth_2x_faster_finetuning_edit.ipynb`: Notebook for fine-tuning the Llama model, with a focus on chat templates and faster finetuning.
  - `Phi_3_Medium_4K_Instruct_Unsloth_2x_faster_finetuning.ipynb`: Another fine-tuning notebook, possibly for a different model or setting.
  - `req_conda_unsloth.txt`: A text file listing Conda environment requirements for the Unsloth Tunes project.
  - `restore_artifacts.ipynb`: A notebook dedicated to restoring or managing artifacts related to the Unsloth Tunes project.

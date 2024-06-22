import dspy
import random
import pandas as pd
# from dotenv import load_dotenv
from dspy.datasets import DataLoader
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, LabeledFewShot
import os
import csv

# Load environment variables
# _ = load_dotenv()

NUM_THREADS = 16

# Define generation arguments
generation_args = {
    "temperature": 0,
    "max_tokens": 500,
    "stop": "\n\n",
    "model_type": "chat",
    "n": 1
}

# Declare lms as global
base_lm = None
evaluator_lm = None

# Base model configurations
model_info_base = [
    {"model": "llama-3-8b-bnb-4bit-synthetic_text_to_sql-lora-3epochs-Q5_K_M:latest", "base_url": 'http://localhost:11435'}
    # Add more base models here as needed
]

# Evaluator model configurations
model_info_eval = [
    {"evaluator_model": "llama3:70b", "evaluator_base_url": 'http://localhost:11434'}
    # Add more evaluator models here as needed
]

# Set random seed
random_seed = 1399
random.seed(random_seed)

def configure_model(model_name, base_url, evaluator_model_name=None, evaluator_base_url=None):
    """Configure and return a local Ollama model."""
    base_lm = dspy.OllamaLocal(model=model_name, base_url=base_url)
    if evaluator_model_name:
        evaluator_model = dspy.OllamaLocal(model=evaluator_model_name, base_url=evaluator_base_url)
        dspy.configure(lm=base_lm)
        return base_lm, evaluator_model
    else:
        dspy.configure(lm=base_lm)
        return base_lm

def load_and_sample_dataset(number_of_samples=200):
    """Load and sample the dataset from HuggingFace."""
    dl = DataLoader()
    testset = dl.from_huggingface(
        dataset_name="gretelai/synthetic_text_to_sql",
        fields=("sql_prompt", "sql_context", "sql"),
        input_keys=("sql_prompt", "sql_context"),
        split="test"
    )
    
    return dl.sample(dataset=testset, n=number_of_samples)

def split_dataset(dataset, train_ratio=0.4, val_ratio=0.2):
    """Split the dataset into training, validation, and test sets."""
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    trainset = dataset[:train_size]
    valset = dataset[train_size:train_size + val_size]
    testset = dataset[train_size + val_size:]

    return trainset, valset, testset

class TextToSql(dspy.Signature):
    sql_prompt = dspy.InputField(desc="Natural language query")
    sql_context = dspy.InputField(desc="Context for the query")
    sql = dspy.OutputField(desc="SQL query")

class Correctness(dspy.Signature):
    sql_prompt = dspy.InputField(desc="Natural language query")
    sql_context = dspy.InputField(desc="Context for the query")
    sql = dspy.InputField(desc="SQL query")
    correct = dspy.OutputField(desc="Indicate whether the SQL query correctly answers the natural language query based on the given context", prefix="Yes/No:")

def correctness_metric(example, pred, trace=None):
    """Evaluate the correctness of the generated SQL query."""
    sql_prompt, sql_context, sql = example.sql_prompt, example.sql_context, pred.sql
    correctness = dspy.Predict(Correctness)

    with dspy.context(lm=evaluator_lm):
        correct = correctness(sql_prompt=sql_prompt, sql_context=sql_context, sql=sql)

    score = int(correct.correct == "Yes")
    return score if trace is None else score == 1

class TextToSqlProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.program = dspy.ChainOfThought(signature=TextToSql)

    def forward(self, sql_prompt, sql_context):
        return self.program(sql_prompt=sql_prompt, sql_context=sql_context)

import time

def evaluate_model(base_lm, evaluator_lm, trainset, valset, testset, model_name, evaluator_model_name, random_seed, run_index=None):
    """Evaluate the model using different optimization techniques and return the results."""
    
    results = {
        "Model": model_name,
        "Evaluator Model": evaluator_model_name,
        "Random Seed": random_seed,
        "Number of Samples": number_of_samples,
        # "Run Index": run_index
    }
    
    generate_sql_query = dspy.Predict(signature=TextToSql)

    total_start_time = time.time()

    # Evaluate on validation set
    start_time = time.time()
    print("Evaluating on validation set")
    evaluate = Evaluate(devset=valset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0, return_all_scores=True, return_outputs=True)
    val_scores, val_results = evaluate(generate_sql_query)
    val_time = round(time.time() - start_time, 4)

    # Evaluate on test set
    start_time = time.time()
    print("Evaluating on test set")
    evaluate = Evaluate(devset=testset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0, return_all_scores=True, return_outputs=True)
    test_scores, test_results = evaluate(generate_sql_query)
    test_time = round(time.time() - start_time, 2)

    # Optimize with LabeledFewShot and evaluate
    start_time = time.time()
    print("Optimizing with LabeledFewShot and evaluating")
    k = 4
    optimizer = LabeledFewShot(k=k)
    optimized_program = optimizer.compile(student=TextToSqlProgram(), trainset=trainset)
    fewshot_optimization_time = round(time.time() - start_time, 2)

    start_time = time.time()
    print("Evaluating optimized program on validation set")
    evaluate = Evaluate(devset=valset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0, return_all_scores=True, return_outputs=True)
    val_optimized_scores, val_optimized_results = evaluate(optimized_program)
    fewshot_val_time = round(time.time() - start_time, 2)

    start_time = time.time()
    print("Evaluating optimized program on test set")
    evaluate = Evaluate(devset=testset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0, return_all_scores=True, return_outputs=True)
    test_optimized_scores, test_optimized_results = evaluate(optimized_program)
    fewshot_test_time = round(time.time() - start_time, 2)

    # Optimize with BootstrapFewShotWithRandomSearch and evaluate
    start_time = time.time()
    print("Optimizing with BootstrapFewShotWithRandomSearch and evaluating")
    max_bootstrapped_demos = 2
    num_candidate_programs = 2
    optimizer2 = BootstrapFewShotWithRandomSearch(metric=correctness_metric, max_bootstrapped_demos=max_bootstrapped_demos, num_candidate_programs=num_candidate_programs, num_threads=NUM_THREADS)
    optimized_program_2 = optimizer2.compile(student=TextToSqlProgram(), trainset=trainset, valset=valset)
    bootstrapfewshot_optimization_time = round(time.time() - start_time, 2)

    start_time = time.time()
    print("Evaluating BootstrapFewShot optimized program on validation set")
    evaluate = Evaluate(devset=valset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0, return_all_scores=True, return_outputs=True)
    val_optimized_scores_2, val_optimized_results_2 = evaluate(optimized_program_2)
    bootstrapfewshot_val_time = round(time.time() - start_time, 2)

    start_time = time.time()
    print("Evaluating BootstrapFewShot optimized program on test set")
    evaluate = Evaluate(devset=testset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0, return_all_scores=True, return_outputs=True)
    test_optimized_scores_2, test_optimized_results_2 = evaluate(optimized_program_2)
    bootstrapfewshot_test_time = round(time.time() - start_time, 2)

    total_time = round(time.time() - total_start_time, 2)

    print("Evaluation complete")
    results.update({
        "Total Time": total_time,
        "Validation Time": val_time,
        "Validation Scores": val_scores,
        "Validation Results": val_results,
        "Test Time": test_time,
        "Test Scores": test_scores,
        "Test Results": test_results,
        "Optimization Time - LabeledFewShot": fewshot_optimization_time,
        "Number of candidate programs - LabeledFewShot": k,
        "Validation Time - LabeledFewShot": fewshot_val_time,
        "Validation Scores - LabeledFewShot": val_optimized_scores,
        "Validation Results - LabeledFewShot": val_optimized_results,
        "Test Time - LabeledFewShot": fewshot_test_time,
        "Test Scores - LabeledFewShot": test_optimized_scores,
        "Test Results - LabeledFewShot": test_optimized_results,
        "Optimization Time - BootstrapFewShot": bootstrapfewshot_optimization_time,
        "Number of candidate programs - BootstrapFewShot": num_candidate_programs,
        "Max Bootstrapped Demos - BootstrapFewShot": max_bootstrapped_demos,
        "Validation Time - BootstrapFewShot": bootstrapfewshot_val_time,
        "Validation Scores - BootstrapFewShot": val_optimized_scores_2,
        "Validation Results - BootstrapFewShot": val_optimized_results_2,
        "Test Time - BootstrapFewShot": bootstrapfewshot_test_time,
        "Test Scores - BootstrapFewShot": test_optimized_scores_2,
        "Test Results - BootstrapFewShot": test_optimized_results_2,
    })

    return results

# Function to serialize columns with complex data
def serialize_complex_columns(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
    return df




# def main():
"""Main function to orchestrate the model evaluation and logging."""
number_of_samples = 20
testset = load_and_sample_dataset(number_of_samples)
trainset, valset, testset = split_dataset(testset)

all_results = []
csv_file = "log_evaluations.csv"
existing_df = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame()

for base_model in model_info_base:
    for eval_model in model_info_eval:     
        base_lm = dspy.OllamaLocal(model=base_model["model"], base_url=base_model["base_url"])
        evaluator_lm = dspy.OllamaLocal(model=eval_model["evaluator_model"], base_url=eval_model["evaluator_base_url"])

        dspy.configure(lm=base_lm)
        
        results = evaluate_model(base_lm, evaluator_lm, trainset, valset, testset, base_model["model"], eval_model["evaluator_model"], random_seed, number_of_samples)#, run_index)
        all_results.append(results)
        
# Check if there are any results in all_results
if all_results:
    log_df = pd.DataFrame(all_results)
    log_df = pd.concat([existing_df, log_df], ignore_index=True) if not existing_df.empty else log_df
    print(log_df)
    # log_df.to_csv(csv_file, index=False)
    # use unique delimiter to avoid conflicts with commas in the data, since data contains sql use writer since i want to have a limiter with more than one character e.v. "|||"
    # Write to CSV with the unique delimiter using csv module
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter="|||")
        writer.writerow(log_df.columns)
        for row in log_df.itertuples(index=False, name=None):
            writer.writerow(row)

else:
    print("No new evaluations were performed.")

# if __name__ == "__main__":
#     main()

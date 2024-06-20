import dspy
import random
import pandas as pd
# from dotenv import load_dotenv
from dspy.datasets import DataLoader
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, LabeledFewShot
import os

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

# def evaluate_model(base_lm, evaluator_lm, trainset, valset, testset, model_name, evaluator_model_name, random_seed, run_index):
#     """Evaluate the model using different optimization techniques and return the results."""
    
#     results = {
#         "Model": model_name,
#         "Evaluator Model": evaluator_model_name,
#         "Random Seed": random_seed,
#         "Run Index": run_index
#     }
    
#     generate_sql_query = dspy.Predict(signature=TextToSql)

#     start_time = time.time()
#     print("Evaluating on validation set")
#     evaluate = Evaluate(devset=valset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0)
#     val_scores = evaluate(generate_sql_query)
#     results["Validation Time"] = time.time() - start_time

#     start_time = time.time()
#     print("Evaluating on test set")
#     evaluate = Evaluate(devset=testset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0)
#     test_scores = evaluate(generate_sql_query)
#     results["Test Time"] = time.time() - start_time

#     start_time = time.time()
#     print("Optimizing with LabeledFewShot and evaluating")
#     optimizer = LabeledFewShot(k=4)
#     optimized_program = optimizer.compile(student=TextToSqlProgram(), trainset=trainset)
#     results["LabeledFewShot Optimization Time"] = time.time() - start_time

#     start_time = time.time()
#     print("Evaluating optimized program on validation set")
#     evaluate = Evaluate(devset=valset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0)
#     val_optimized_scores = evaluate(optimized_program)
#     results["LabeledFewShot Validation Time"] = time.time() - start_time

#     start_time = time.time()
#     print("Evaluating optimized program on test set")
#     evaluate = Evaluate(devset=testset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0)
#     test_optimized_scores = evaluate(optimized_program)
#     results["LabeledFewShot Test Time"] = time.time() - start_time

#     start_time = time.time()
#     print("Optimizing with BootstrapFewShotWithRandomSearch and evaluating")
#     optimizer2 = BootstrapFewShotWithRandomSearch(metric=correctness_metric, max_bootstrapped_demos=2, num_candidate_programs=2, num_threads=NUM_THREADS)
#     optimized_program_2 = optimizer2.compile(student=TextToSqlProgram(), trainset=trainset, valset=valset)
#     results["BootstrapFewShot Optimization Time"] = time.time() - start_time

#     start_time = time.time()
#     print("Evaluating BootstrapFewShot optimized program on validation set")
#     evaluate = Evaluate(devset=valset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0)
#     val_optimized_scores_2 = evaluate(optimized_program_2)
#     results["BootstrapFewShot Validation Time"] = time.time() - start_time

#     start_time = time.time()
#     print("Evaluating BootstrapFewShot optimized program on test set")
#     evaluate = Evaluate(devset=testset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0)
#     test_optimized_scores_2 = evaluate(optimized_program_2)
#     results["BootstrapFewShot Test Time"] = time.time() - start_time

#     print("Evaluation complete")
#     results.update({
#         "Validation Scores": val_scores,
#         "Test Scores": test_scores,
#         "Validation Scores (FewShot)": val_optimized_scores,
#         "Test Scores (FewShot)": test_optimized_scores,
#         "Validation Scores (BootstrapFewShot)": val_optimized_scores_2,
#         "Test Scores (BootstrapFewShot)": test_optimized_scores_2
#     })

#     return results

def evaluate_model(base_lm, evaluator_lm, trainset, valset, testset, model_name, evaluator_model_name, random_seed, run_index):
    """Evaluate the model using different optimization techniques and return the results."""
    
    results = {
        "Model": model_name,
        "Evaluator Model": evaluator_model_name,
        "Random Seed": random_seed,
        "Run Index": run_index
    }
    
    generate_sql_query = dspy.Predict(signature=TextToSql)

    # Evaluate on validation set
    start_time = time.time()
    print("Evaluating on validation set")
    evaluate = Evaluate(devset=valset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0, return_all_scores=True, return_outputs=True)
    val_evaluation = evaluate(generate_sql_query)
    if len(val_evaluation) == 3:
        val_scores, val_results, val_individual_scores = val_evaluation
    else:
        val_scores, val_results = val_evaluation
        val_individual_scores = None
    results["Validation Time"] = time.time() - start_time

    # Evaluate on test set
    start_time = time.time()
    print("Evaluating on test set")
    evaluate = Evaluate(devset=testset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0, return_all_scores=True, return_outputs=True)
    test_evaluation = evaluate(generate_sql_query)
    if len(test_evaluation) == 3:
        test_scores, test_results, test_individual_scores = test_evaluation
    else:
        test_scores, test_results = test_evaluation
        test_individual_scores = None
    results["Test Time"] = time.time() - start_time

    # Optimize with LabeledFewShot and evaluate
    start_time = time.time()
    print("Optimizing with LabeledFewShot and evaluating")
    optimizer = LabeledFewShot(k=4)
    optimized_program = optimizer.compile(student=TextToSqlProgram(), trainset=trainset)
    results["LabeledFewShot Optimization Time"] = time.time() - start_time

    start_time = time.time()
    print("Evaluating optimized program on validation set")
    evaluate = Evaluate(devset=valset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0, return_all_scores=True, return_outputs=True)
    val_optimized_evaluation = evaluate(optimized_program)
    if len(val_optimized_evaluation) == 3:
        val_optimized_scores, val_optimized_results, val_optimized_individual_scores = val_optimized_evaluation
    else:
        val_optimized_scores, val_optimized_results = val_optimized_evaluation
        val_optimized_individual_scores = None
    results["LabeledFewShot Validation Time"] = time.time() - start_time

    start_time = time.time()
    print("Evaluating optimized program on test set")
    evaluate = Evaluate(devset=testset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0, return_all_scores=True, return_outputs=True)
    test_optimized_evaluation = evaluate(optimized_program)
    if len(test_optimized_evaluation) == 3:
        test_optimized_scores, test_optimized_results, test_optimized_individual_scores = test_optimized_evaluation
    else:
        test_optimized_scores, test_optimized_results = test_optimized_evaluation
        test_optimized_individual_scores = None
    results["LabeledFewShot Test Time"] = time.time() - start_time

    # Optimize with BootstrapFewShotWithRandomSearch and evaluate
    start_time = time.time()
    print("Optimizing with BootstrapFewShotWithRandomSearch and evaluating")
    optimizer2 = BootstrapFewShotWithRandomSearch(metric=correctness_metric, max_bootstrapped_demos=2, num_candidate_programs=2, num_threads=NUM_THREADS)
    optimized_program_2 = optimizer2.compile(student=TextToSqlProgram(), trainset=trainset, valset=valset)
    results["BootstrapFewShot Optimization Time"] = time.time() - start_time

    start_time = time.time()
    print("Evaluating BootstrapFewShot optimized program on validation set")
    evaluate = Evaluate(devset=valset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0, return_all_scores=True, return_outputs=True)
    val_optimized_evaluation_2 = evaluate(optimized_program_2)
    if len(val_optimized_evaluation_2) == 3:
        val_optimized_scores_2, val_optimized_results_2, val_optimized_individual_scores_2 = val_optimized_evaluation_2
    else:
        val_optimized_scores_2, val_optimized_results_2 = val_optimized_evaluation_2
        val_optimized_individual_scores_2 = None
    results["BootstrapFewShot Validation Time"] = time.time() - start_time

    start_time = time.time()
    print("Evaluating BootstrapFewShot optimized program on test set")
    evaluate = Evaluate(devset=testset, metric=correctness_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0, return_all_scores=True, return_outputs=True)
    test_optimized_evaluation_2 = evaluate(optimized_program_2)
    if len(test_optimized_evaluation_2) == 3:
        test_optimized_scores_2, test_optimized_results_2, test_optimized_individual_scores_2 = test_optimized_evaluation_2
    else:
        test_optimized_scores_2, test_optimized_results_2 = test_optimized_evaluation_2
        test_optimized_individual_scores_2 = None
    results["BootstrapFewShot Test Time"] = time.time() - start_time

    print("Evaluation complete")
    results.update({
        "Validation Scores": val_scores,
        "Validation Results": val_results,
        "Validation Individual Scores": val_individual_scores,
        "Test Scores": test_scores,
        "Test Results": test_results,
        "Test Individual Scores": test_individual_scores,
        "Validation Scores (FewShot)": val_optimized_scores,
        "Validation Results (FewShot)": val_optimized_results,
        "Validation Individual Scores (FewShot)": val_optimized_individual_scores,
        "Test Scores (FewShot)": test_optimized_scores,
        "Test Results (FewShot)": test_optimized_results,
        "Test Individual Scores (FewShot)": test_optimized_individual_scores,
        "Validation Scores (BootstrapFewShot)": val_optimized_scores_2,
        "Validation Results (BootstrapFewShot)": val_optimized_results_2,
        "Validation Individual Scores (BootstrapFewShot)": val_optimized_individual_scores_2,
        "Test Scores (BootstrapFewShot)": test_optimized_scores_2,
        "Test Results (BootstrapFewShot)": test_optimized_results_2,
        "Test Individual Scores (BootstrapFewShot)": test_optimized_individual_scores_2
    })

    return results





# def main():
"""Main function to orchestrate the model evaluation and logging."""
testset = load_and_sample_dataset(number_of_samples=20)
trainset, valset, testset = split_dataset(testset)

all_results = []
csv_file = "log_evaluations.csv"
existing_df = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame()

for base_model in model_info_base:
    for eval_model in model_info_eval:
        # model_name = base_model["model"]
        # evaluator_model_name = eval_model["evaluator_model"]
        

        # Check if the existing DataFrame is not empty
        if not existing_df.empty:
            # Filter the DataFrame for rows where the 'Model' matches 'model_name',
            # 'Evaluator Model' matches 'evaluator_model_name', and 'Random Seed' matches 'random_seed'.
            # Then, find the maximum 'Run Index' among these filtered rows.
            max_run_index = existing_df[(existing_df["Model"] == base_model["model"]) & 
                                        (existing_df["Evaluator Model"] == eval_model["evaluator_model"]) & 
                                        (existing_df["Random Seed"] == random_seed)]["Run Index"].max()
            
            # If 'max_run_index' is NaN (which means no matching rows were found), set 'run_index' to 1.
            # Otherwise, increment 'max_run_index' by 1 to get the new run index.
            # This ensures that each run has a unique index based on the filter criteria.
            run_index = 1 if pd.isna(max_run_index) else max_run_index + 1
        else:
            # If the existing DataFrame is empty, this must be the first run.
            # Therefore, set 'run_index' to 1.
            run_index = 1
            
        
        # base_lm, evaluator_lm = configure_model(base_model["model"], base_model["base_url"], eval_model["evaluator_model"], eval_model["evaluator_base_url"])
        
        base_lm = dspy.OllamaLocal(model=base_model["model"], base_url=base_model["base_url"])
        evaluator_lm = dspy.OllamaLocal(model=eval_model["evaluator_model"], base_url=eval_model["evaluator_base_url"])

        dspy.configure(lm=base_lm)
        
        results = evaluate_model(base_lm, evaluator_lm, trainset, valset, testset, base_model["model"], eval_model["evaluator_model"], random_seed, run_index)
        all_results.append(results)
        
# Check if there are any results in all_results
if all_results:
    log_df = pd.DataFrame(all_results)
    log_df = pd.concat([existing_df, log_df], ignore_index=True) if not existing_df.empty else log_df
    print(log_df)
    log_df.to_csv(csv_file, index=False)
else:
    print("No new evaluations were performed.")

# if __name__ == "__main__":
#     main()

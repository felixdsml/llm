"""
This script evaluates different language models on a Text-to-SQL task using the DSPy framework.
It configures base and evaluator models, loads and splits datasets, evaluates models using various metrics,
and logs the results to an Excel file. The evaluation process includes optimization using LabeledFewShot and BootstrapFewShotWithRandomSearch.
"""

import os
import re
import time
import random
import pandas as pd
import dspy
import phoenix as px

from dspy import evaluate
from dspy.datasets import DataLoader
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, LabeledFewShot
# from utils_random_search import BootstrapFewShotWithRandomSearch
from utils_evaluate import Evaluate as Evaluate_multiple
from dspy.teleprompt import LabeledFewShot
from openinference.semconv.resource import ResourceAttributes
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference.semconv.trace import SpanAttributes
from phoenix.trace import using_project

# Configure logging with Phoenix
# Phoenix is used for tracing and monitoring the evaluation process
endpoint = "http://127.0.0.1:6006/v1/traces"
resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
span_otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=span_otlp_exporter))
trace_api.set_tracer_provider(tracer_provider=tracer_provider)
DSPyInstrumentor().instrument()

NUM_THREADS = 16

# Set debug flag
IF_DEBUG = False

# Number of samples to generate for evaluation
number_of_samples = 200

# Set random seed for reproducibility
random_seed = 1
random.seed(random_seed)

# Configuration for base and evaluator models
model_info_base = [
    {"model": "mistral:7b-instruct-v0.3-q5_K_M", "base_url": 'http://localhost:11435'},
    {"model": "llama-3-8b-bnb-4bit-synthetic_text_to_sql-lora-3epochs-Q5_K_M:latest", "base_url": 'http://localhost:11435'},
    {"model": "llama-3-8b-Instruct-bnb-4bit-synthetic_text_to_sql-lora-3epochs-Q5_K_M:latest", "base_url": 'http://localhost:11435'} ,
    {"model": "Phi-3-medium-4k-instruct-synthetic_text_to_sql-lora-3epochs-q5_k_m:latest", "base_url": 'http://localhost:11435'},
    {"model": "phi3:14b-medium-4k-instruct-q5_K_M", "base_url": 'http://localhost:11435'}, 
    {"model": "llama3:8b-text-q5_K_M", "base_url": 'http://localhost:11435'},
    {"model": "llama3:8b-instruct-q5_K_M", "base_url": 'http://localhost:11435'},
    {"model": "command-r", "base_url": 'http://localhost:11435'},
    {"model": "codegemma:7b-code-q5_K_M", "base_url": 'http://localhost:11435'},
    {"model": "aya:35b", "base_url": 'http://localhost:11435'},
    {"model": "qwen2:7b-instruct-q5_K_M", "base_url": 'http://localhost:11435'},
    # {"model": "deepseek-coder-v2:16b-lite-instruct-q5_K_M", "base_url": 'http://localhost:11435'}, TypeError: unsupported operand type(s) for +=: 'int' and 'NoneType'
    {"model": "llama3:8b-instruct-fp16", "base_url": 'http://localhost:11435'},
    {"model": "codegemma:7b-code-fp16", "base_url": 'http://localhost:11435'},
    # Add more base models here as needed
]

model_info_eval = [
    {"evaluator_model": "llama3:70b", "evaluator_base_url": 'http://localhost:11434'}
]


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

def debug_testset(dataset):
    """For testing purposes, return 5 samples from each set."""
    train_size = 2
    val_size = 2
    test_size = 2

    trainset = dataset[:train_size]
    valset = dataset[train_size:train_size + val_size]
    testset = dataset[train_size + val_size:train_size + val_size + test_size]
    
    print("len of sets: ", len(trainset), len(valset), len(testset))

    return trainset, valset, testset

def save_large_result(result, model_name, evaluator_model_name, set_type, seed, sample_size):
    """Save large results to a text file in a subfolder and return the filename, adding an index if the file already exists."""
    os.makedirs("logs", exist_ok=True)
    model_name = model_name.replace(":", "_").replace("-", "_")
    evaluator_model_name = evaluator_model_name.replace(":", "_").replace("-", "_")
    base_filename = f"{model_name}_{evaluator_model_name}_{seed}_{sample_size}_{set_type}"
    filename = os.path.join("logs", f"{base_filename}.txt")
    index = 1
    while os.path.exists(filename):
        filename = os.path.join("logs", f"{base_filename}_{index}.txt")
        index += 1
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(str(result))
    return filename

def save_optimized_program(optimized_program, model_name, evaluator_model_name, set_type, seed, sample_size):
    """Save the optimized program to a file in the 'optimized_programs' directory, adding an index if the file already exists."""
    os.makedirs("optimized_programs", exist_ok=True)
    model_name = model_name.replace(":", "_").replace("-", "_")
    evaluator_model_name = evaluator_model_name.replace(":", "_").replace("-", "_")
    base_filename = f"{model_name}_{evaluator_model_name}_{seed}_{sample_size}_{set_type}"
    filename = os.path.join("optimized_programs", f"{base_filename}.json")
    index = 1
    while os.path.exists(filename):
        filename = os.path.join("optimized_programs", f"{base_filename}_{index}.json")
        index += 1
    optimized_program.save(filename)
    return filename

class TextToSql(dspy.Signature):
    """Signature for Text to SQL generation task."""
    sql_prompt = dspy.InputField(desc="Natural language query")
    sql_context = dspy.InputField(desc="Context for the query")
    sql = dspy.OutputField(desc="SQL query")

class SQLMatch(dspy.Signature):
    """Signature for matching SQL queries."""
    sql_reference = dspy.InputField(desc="Reference SQL query")
    sql_predicted = dspy.InputField(desc="Predicted SQL query")
    match = dspy.OutputField(desc="Indicate whether the reference and predicted SQL query match", prefix="Yes/No:")

match_instruction = """
Given a reference SQL query and a predicted SQL query, determine if the predicted SQL query matches the reference SQL query exactly. Output only 'Yes' if it matches, otherwise output only 'No'.
"""
SQLMatch = SQLMatch.with_instructions(match_instruction)

class SQLCorrectness(dspy.Signature):
    """Signature for evaluating the correctness of SQL queries."""
    sql_prompt = dspy.InputField(desc="Natural language query")
    sql_context = dspy.InputField(desc="Context for the query")
    sql_predicted = dspy.InputField(desc="Predicted SQL query")
    correct = dspy.OutputField(desc="Indicate whether the predicted SQL query correctly answers the natural language query based on the given context. Output only 'Yes' if it is correct, otherwise output only 'No'", prefix="Yes/No:")

correctness_instruction = """
Given a natural language query, its context, and a predicted SQL query, determine if the predicted SQL query correctly answers the natural language query based on the context.
"""
SQLCorrectness = SQLCorrectness.with_instructions(correctness_instruction)

class SQLExecutable(dspy.Signature):
    """Signature for evaluating if the SQL query is executable."""
    sql_reference = dspy.InputField(desc="Reference SQL query")
    sql_predicted = dspy.InputField(desc="Predicted SQL query")
    executable = dspy.OutputField(desc="Indicate whether the predicted SQL query is executable. Output only 'Yes' if it is correct, otherwise output only 'No'", prefix="Yes/No:")

executable_instruction = """
Evaluate the provided SQL query to determine if it is executable as-is, without any extraneous text, rationale, or errors. 

Criteria:
1. The SQL query must not include any additional text such as rationale, prompts, or context.
2. It must be executable without modifications.

Example which does not meet the criteria:

Reference SQL query: "SELECT subscriber_id, speed, month FROM (SELECT subscriber_id, speed, month, LAG(speed, 1) OVER (PARTITION BY subscriber_id ORDER BY month) as prev_speed, LAG(speed, 2) OVER (PARTITION BY subscriber_id ORDER BY month) as prev_prev_speed FROM mobile_usage_detailed) t WHERE t.speed < 0.75 * t.prev_speed AND t.speed < 0.75 * t.prev_prev_speed ORDER BY subscriber_id;"

Predicted SQL query: "Here is the completed signature for the Text-to-SQL generation task:
Sql Prompt: Find the mobile subscribers with consecutive speed drops greater than 25% for the last 3 months, ordered by subscription IDs. 
SELECT subscriber_id, speed, month FROM (SELECT subscriber_id, speed, month, LAG(speed, 1) OVER (PARTITION BY subscriber_id ORDER BY month) as prev_speed, LAG(speed, 2) OVER (PARTITION BY subscriber_id ORDER BY month) as prev_prev_speed FROM mobile_usage_detailed) t WHERE t.speed < 0.75 * t.prev_speed AND t.speed < 0.75 * t.prev_prev_speed ORDER BY subscriber_id;
"""
SQLExecutable = SQLExecutable.with_instructions(executable_instruction)

class TextToSqlProgram(dspy.Module):
    """A module that represents the program for generating SQL from natural language."""
    def __init__(self):
        super().__init__()
        self.program = dspy.ChainOfThought(signature=TextToSql)

    def forward(self, sql_prompt, sql_context):
        # current_span = trace_api.get_current_span()
        return self.program(sql_prompt=sql_prompt, sql_context=sql_context)#, span_id=current_span.get_span_context().span_id)
    

def match_metric(example, pred, trace=None, span_id=None):
    """Evaluate if the predicted SQL query matches the reference SQL query."""
    sql_reference, sql_predicted = example.sql, pred.sql
    match = dspy.Predict(SQLMatch)
    with dspy.context(lm=evaluator_lm):
        is_match = match(sql_reference=sql_reference, sql_predicted=sql_predicted)
    match_output = is_match.match.strip()
    match_score = int(re.search(r'\bYes\b', match_output, re.IGNORECASE) is not None)
    return match_score

def correctness_metric(example, pred, trace=None):
    """Evaluate if the predicted SQL query correctly answers the natural language query."""
    sql_prompt, sql_context, sql_predicted = example.sql_prompt, example.sql_context, pred.sql
    correctness = dspy.Predict(SQLCorrectness)
    with dspy.context(lm=evaluator_lm):
        is_correct = correctness(sql_prompt=sql_prompt, sql_context=sql_context, sql_predicted=sql_predicted)
    correct_output = is_correct.correct.strip()
    correct_score = int(re.search(r'\bYes\b', correct_output, re.IGNORECASE) is not None)
    
    return correct_score


def executable_metric(example, pred, trace=None):
    """Evaluate if the predicted SQL query is executable."""
    sql_predicted = pred.sql
    executable = dspy.Predict(SQLExecutable)
    with dspy.context(lm=evaluator_lm):
        is_executable = executable(sql_predicted=sql_predicted)
    executable_output = is_executable.executable.strip()
    executable_score = int(re.search(r'\bYes\b', executable_output, re.IGNORECASE) is not None)
    
    return executable_score
F

# Combined metric function
def combined_metric(example, pred, trace=None):
    """Evaluate the match, correctness, and executability of the predicted SQL query."""
    sql_reference, sql_predicted = example.sql, pred.sql
    sql_prompt, sql_context = example.sql_prompt, example.sql_context
    
    match = dspy.Predict(SQLMatch)
    correctness = dspy.Predict(SQLCorrectness)
    executable = dspy.Predict(SQLExecutable)
    
    with dspy.context(lm=evaluator_lm):
        is_match = match(sql_reference=sql_reference, sql_predicted=sql_predicted)
        is_correct = correctness(sql_prompt=sql_prompt, sql_context=sql_context, sql_predicted=sql_predicted)
        is_executable = executable(sql_predicted=sql_predicted)
        
    match_output = is_match.match.strip()
    correct_output = is_correct.correct.strip()
    executable_output = is_executable.executable.strip()
    
    match_score = int(re.search(r'\bYes\b', match_output, re.IGNORECASE) is not None)
    correct_score = int(re.search(r'\bYes\b', correct_output, re.IGNORECASE) is not None)
    executable_score = int(re.search(r'\bYes\b', executable_output, re.IGNORECASE) is not None)
    
    score = (executable_score << 2) | (match_score << 1) | correct_score
    return score / 7  # Normalize to a score between 0 and 1



def evaluate_model(base_lm, evaluator_lm, trainset, valset, testset, model_name, evaluator_model_name, random_seed, run_index=None):
    """Evaluate the model using different optimization techniques and return the results."""
    
    def evaluate_set(devset, program, label):
        """Evaluate a given set with the specified program."""
        print(f"Evaluating on {label} set")
        start_time = time.time()
     
        # Define the metrics
        metrics = [match_metric, correctness_metric, executable_metric]

        # Evaluate all metrics
        evaluate = Evaluate_multiple(
            devset=devset,
            metrics=metrics,
            num_threads=NUM_THREADS,
            display_progress=True,
            display_table=0,
            return_all_scores=True,
            return_outputs=True
        )

        avg_metrics, results = evaluate(program)

        # Extract individual scores if needed
        match_score = avg_metrics[match_metric.__name__]
        correct_score = avg_metrics[correctness_metric.__name__]
        executable_score = avg_metrics[executable_metric.__name__]

        # Extract individual results if needed
        match_result = [(example, prediction, scores[0]) for example, prediction, scores in results]
        correct_result = [(example, prediction, scores[1]) for example, prediction, scores in results]
        executable_result = [(example, prediction, scores[2]) for example, prediction, scores in results]

        
        # Combine the scores
        combined_score = ((int(match_score) << 2) | (int(correct_score) << 1) | int(executable_score)) / 7
        eval_time = round(time.time() - start_time, 2)
        
        return match_score, correct_score, executable_score, combined_score, match_result, correct_result, executable_result, eval_time

    def optimize_and_evaluate(optimizer, trainset, valset, testset, program_label):
        """Optimize the program and evaluate on validation and test sets."""
        start_time = time.time()
        print(f"Optimizing with {program_label} and evaluating")
        if program_label == "LabeledFewShot":
            optimized_program = optimizer.compile(student=TextToSqlProgram(), trainset=trainset)
        else:
            optimized_program = optimizer.compile(student=TextToSqlProgram(), trainset=trainset, valset=valset)
        optimization_time = round(time.time() - start_time, 2)
        save_optimized_program(optimized_program, model_name, evaluator_model_name, program_label, random_seed, number_of_samples)
        
        test_match_scores, test_correct_scores, test_executable_scores, test_combined_scores, test_match_results, test_correct_results, test_executable_results, test_time = evaluate_set(testset, optimized_program, f"{program_label} test")
        
        return (test_match_scores, test_correct_scores, test_executable_scores, test_combined_scores, test_match_results, test_correct_results, test_executable_results, test_time, optimization_time)

    results = {
        "Model": model_name,
        "Evaluator Model": evaluator_model_name,
        "Random Seed": random_seed,
        "Number of Samples": number_of_samples,
    }
    
    generate_sql_query = dspy.Predict(signature=TextToSql)
    total_start_time = time.time()
    
    # # Evaluate on validation and test sets
  
    test_match_scores, test_correct_scores, test_executable_scores, test_combined_scores, test_match_results, test_correct_results, test_executable_results, test_time = evaluate_set(testset, generate_sql_query, "test")
    
    # # Optimize with LabeledFewShot and evaluate
    labeled_fewshot_optimizer = LabeledFewShot(k=4)
    (test_fewshot_match_scores, test_fewshot_correct_scores, test_fewshot_executable_scores, test_fewshot_combined_scores, test_fewshot_match_results, test_fewshot_correct_results, test_fewshot_executable_results, test_fewshot_time, 
     fewshot_optimization_time) = optimize_and_evaluate(labeled_fewshot_optimizer, trainset, valset, testset, "LabeledFewShot")
    
    # # Optimize with BootstrapFewShotWithRandomSearch and evaluate
    max_bootstrapped_demos = 2
    num_candidate_programs = 2
    bootstrap_optimizer = BootstrapFewShotWithRandomSearch(metric=combined_metric, max_bootstrapped_demos=max_bootstrapped_demos, num_candidate_programs=num_candidate_programs, num_threads=NUM_THREADS, teacher_settings=dict(lm=evaluator_lm))
    (     test_bootstrap_match_scores, test_bootstrap_correct_scores, test_bootstrap_executable_scores, test_bootstrap_combined_scores, test_bootstrap_match_results, test_bootstrap_correct_results, test_bootstrap_executable_results, test_bootstrap_time, 
     bootstrap_optimization_time) = optimize_and_evaluate(bootstrap_optimizer, trainset, valset, testset, "BootstrapFewShot")
    
    total_time = round(time.time() - total_start_time, 2)
    print("Evaluation complete")

    results.update({
        "Total Time": total_time,
        "Test Match Time": test_time,
        "Test Match Scores": test_match_scores,
        "Test Match Results": save_large_result(test_match_results, model_name, evaluator_model_name, "test_match", random_seed, number_of_samples),
        "Test Correctness Time": test_time,
        "Test Correctness Scores": test_correct_scores,
        "Test Correctness Results": save_large_result(test_correct_results, model_name, evaluator_model_name, "test_correct", random_seed, number_of_samples),
        "Test Executable Time": test_time,
        "Test Executable Scores": test_executable_scores,
        "Test Executable Results": save_large_result(test_executable_results, model_name, evaluator_model_name, "test_executable", random_seed, number_of_samples),
        "Test Combined Scores": test_combined_scores,
        "Optimization Time - LabeledFewShot": fewshot_optimization_time,
        "Test Match Time - LabeledFewShot": test_fewshot_time,
        "Test Match Scores - LabeledFewShot": test_fewshot_match_scores,
        "Test Match Results - LabeledFewShot": save_large_result(test_fewshot_match_results, model_name, evaluator_model_name, "test_fewshot_match", random_seed, number_of_samples),
        "Test Correctness Time - LabeledFewShot": test_fewshot_time,
        "Test Correctness Scores - LabeledFewShot": test_fewshot_correct_scores,
        "Test Correctness Results - LabeledFewShot": save_large_result(test_fewshot_correct_results, model_name, evaluator_model_name, "test_fewshot_correct", random_seed, number_of_samples),
        "Test Executable Time - LabeledFewShot": test_fewshot_time,
        "Test Executable Scores - LabeledFewShot": test_fewshot_executable_scores,
        "Test Executable Results - LabeledFewShot": save_large_result(test_fewshot_executable_results, model_name, evaluator_model_name, "test_fewshot_executable", random_seed, number_of_samples),
        "Test Combined Scores - LabeledFewShot": test_fewshot_combined_scores,
        "Optimization Time - BootstrapFewShot": bootstrap_optimization_time,
        "Test Match Time - BootstrapFewShot": test_bootstrap_time,
        "Test Match Scores - BootstrapFewShot": test_bootstrap_match_scores,
        "Test Match Results - BootstrapFewShot": save_large_result(test_bootstrap_match_results, model_name, evaluator_model_name, "test_bootstrap_match", random_seed, number_of_samples), 
        "Test Correctness Time - BootstrapFewShot": test_bootstrap_time,
        "Test Correctness Scores - BootstrapFewShot": test_bootstrap_correct_scores,
        "Test Correctness Results - BootstrapFewShot": save_large_result(test_bootstrap_correct_results, model_name, evaluator_model_name, "test_bootstrap_correct", random_seed, number_of_samples),
        "Test Executable Time - BootstrapFewShot": test_bootstrap_time,
        "Test Executable Scores - BootstrapFewShot": test_bootstrap_executable_scores,
        "Test Executable Results - BootstrapFewShot": save_large_result(test_bootstrap_executable_results, model_name, evaluator_model_name, "test_bootstrap_executable", random_seed, number_of_samples),
        "Test Combined Scores - BootstrapFewShot": test_bootstrap_combined_scores,
        "Max Bootstrapped Demos": max_bootstrapped_demos,
        "Number of Candidate Programs": num_candidate_programs,
    })

    return results


# Main function to orchestrate the model evaluation and logging
testset = load_and_sample_dataset(number_of_samples)
if IF_DEBUG:
    trainset, valset, testset = debug_testset(testset)
else:
    trainset, valset, testset = split_dataset(testset)

excel_file = "log_evaluations.xlsx"

for base_model in model_info_base:
    for eval_model in model_info_eval:     
        base_lm = dspy.OllamaLocal(model=base_model["model"], base_url=base_model["base_url"])
        evaluator_lm = dspy.OllamaLocal(model=eval_model["evaluator_model"], base_url=eval_model["evaluator_base_url"])
        
        model_name = base_model["model"].replace(":", "_").replace("-", "_").replace(".", "_")
        evaluator_model_name = eval_model["evaluator_model"].replace(":", "_").replace("-", "_").replace(".", "_")
        
        dspy.configure(lm=base_lm)
        with using_project(f'{model_name}_{evaluator_model_name}_{random_seed}_{number_of_samples}'):
            results = evaluate_model(base_lm, evaluator_lm, trainset, valset, testset, base_model["model"], eval_model["evaluator_model"], random_seed, number_of_samples)
        
        all_results = []
        all_results.append(results)
        
        existing_df = pd.read_excel(excel_file) if os.path.exists(excel_file) else pd.DataFrame()
        log_df = pd.DataFrame(all_results)
        log_df = pd.concat([existing_df, log_df], ignore_index=True) if not existing_df.empty else log_df
        log_df.to_excel(excel_file, index=False)
        
        print("Finished evaluation for model: ", base_model["model"], " and evaluator: ", eval_model["evaluator_model"])

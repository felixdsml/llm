import dspy
import random
import pandas as pd
# from dotenv import load_dotenv
from dspy.datasets import DataLoader
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, LabeledFewShot
import os

# Load environment variables
# load_dotenv()

# Define generation arguments
generation_args = {
    "temperature": 0,
    "max_tokens": 500,
    "stop": "\n\n",
    "model_type": "chat",
    "n": 1
}

# Model configurations
model_info = [
    {"model": "llama-3-8b-bnb-4bit-synthetic_text_to_sql-lora-3epochs-Q5_K_M:latest", "base_url": 'http://localhost:11435', "evaluator_model": "llama3:70b", "evaluator_base_url": 'http://localhost:11434'},
    # Add more models here as needed
]

# Set random seed
random_seed = 1399
random.seed(random_seed)1

def configure_model(model_name, base_url):
    """Configure and return a local Ollama model."""
    lm = dspy.OllamaLocal(model=model_name, base_url=base_url)
    dspy.configure(lm=lm)
    return lm

def load_and_sample_dataset():
    """Load and sample the dataset from HuggingFace."""
    dl = DataLoader()
    testset = dl.from_huggingface(
        dataset_name="gretelai/synthetic_text_to_sql",
        fields=("sql_prompt", "sql_context", "sql"),
        input_keys=("sql_prompt", "sql_context"),
        split="test"
    )
    return dl.sample(dataset=testset, n=200)

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

def evaluate_model(lm, evaluator_lm, trainset, valset, testset, model_name, evaluator_model_name, random_seed, run_index):
    """Evaluate the model using different optimization techniques and return the results."""
    dspy.configure(lm=lm)
    generate_sql_query = dspy.Predict(signature=TextToSql)

    # Evaluate on validation set
    evaluate = Evaluate(devset=valset, metric=correctness_metric, num_threads=10, display_progress=True, display_table=0)
    val_scores = evaluate(generate_sql_query)

    # Evaluate on test set
    evaluate = Evaluate(devset=testset, metric=correctness_metric, num_threads=10, display_progress=True, display_table=0)
    test_scores = evaluate(generate_sql_query)

    # Optimize with LabeledFewShot and evaluate
    optimizer = LabeledFewShot(k=4)
    optimized_program = optimizer.compile(student=TextToSqlProgram(), trainset=trainset)

    evaluate = Evaluate(devset=valset, metric=correctness_metric, num_threads=10, display_progress=True, display_table=0)
    val_optimized_scores = evaluate(optimized_program)

    evaluate = Evaluate(devset=testset, metric=correctness_metric, num_threads=10, display_progress=True, display_table=0)
    test_optimized_scores = evaluate(optimized_program)

    # Optimize with BootstrapFewShotWithRandomSearch and evaluate
    optimizer2 = BootstrapFewShotWithRandomSearch(metric=correctness_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=5)
    optimized_program_2 = optimizer2.compile(student=TextToSqlProgram(), trainset=trainset, valset=valset)

    evaluate = Evaluate(devset=valset, metric=correctness_metric, num_threads=10, display_progress=True, display_table=0)
    val_optimized_scores_2 = evaluate(optimized_program_2)

    evaluate = Evaluate(devset=testset, metric=correctness_metric, num_threads=10, display_progress=True, display_table=0)
    test_optimized_scores_2 = evaluate(optimized_program_2)

    return {
        "Model": model_name,
        "Evaluator Model": evaluator_model_name,
        "Random Seed": random_seed,
        "Run Index": run_index,
        "Validation Scores": val_scores,
        "Test Scores": test_scores,
        "Validation Scores (FewShot)": val_optimized_scores,
        "Test Scores (FewShot)": test_optimized_scores,
        "Validation Scores (BootstrapFewShot)": val_optimized_scores_2,
        "Test Scores (BootstrapFewShot)": test_optimized_scores_2
    }



def main():
    """Main function to orchestrate the model evaluation and logging."""
    testset = load_and_sample_dataset()
    trainset, valset, testset = split_dataset(testset)

    all_results = []
    csv_file = "log_evaluations.csv"
    existing_df = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame()

    for model in model_info:
        model_name = model["model"]
        evaluator_model_name = model["evaluator_model"]

        if not existing_df.empty:
            # Calculate the new run index
            max_run_index = existing_df[(existing_df["Model"] == model_name) & 
                                        (existing_df["Evaluator Model"] == evaluator_model_name) & 
                                        (existing_df["Random Seed"] == random_seed)]["Run Index"].max()
            run_index = 1 if pd.isna(max_run_index) else max_run_index + 1
        else:
            run_index = 1

        lm = configure_model(model["model"], model["base_url"])
        evaluator_lm = configure_model(model["evaluator_model"], model["evaluator_base_url"])
        results = evaluate_model(lm, evaluator_lm, trainset, valset, testset, model["model"], model["evaluator_model"], random_seed, run_index)
        all_results.append(results)

    if all_results:
        log_df = pd.DataFrame(all_results)
        log_df = pd.concat([existing_df, log_df], ignore_index=True) if not existing_df.empty else log_df
        print(log_df)
        log_df.to_csv(csv_file, index=False)
    else:
        print("No new evaluations were performed.")

if __name__ == "__main__":
    main()

import dspy
import random
import pandas as pd
from dotenv import load_dotenv
from dspy.datasets import DataLoader
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, LabeledFewShot

_ = load_dotenv()

generation_args = {
    "temperature": 0,
    "max_tokens": 500,
    "stop": "\n\n",
    "model_type": "chat",
    "n": 1
}

model_info = [
    {"model": "llama-3-8b-bnb-4bit-synthetic_text_to_sql-lora-3epochs-Q5_K_M:latest", "base_url": 'http://localhost:11435'},
    {"model": "llama3:70b", "base_url": 'http://localhost:11434'},
    # Add more models here as needed
]

def configure_model(model_name, base_url):
    lm = dspy.OllamaLocal(model=model_name, base_url=base_url)
    dspy.configure(lm=lm)
    return lm

random.seed(1399)

dl = DataLoader()

testset = dl.from_huggingface(
    dataset_name="gretelai/synthetic_text_to_sql",
    fields=("sql_prompt", "sql_context", "sql"),
    input_keys=("sql_prompt", "sql_context"),
    split="test"
)

testset = dl.sample(dataset=testset, n=200)

total_size = len(testset)
train_size = int(total_size * 0.4)
val_size = int(total_size * 0.2)

trainset = testset[:train_size]
valset = testset[train_size:train_size + val_size]
testset = testset[train_size + val_size:]

class TextToSql(dspy.Signature):
    sql_prompt = dspy.InputField(desc="Natural language query")
    sql_context = dspy.InputField(desc="Context for the query")
    sql = dspy.OutputField(desc="SQL query")

class Correctness(dspy.Signature):
    sql_prompt = dspy.InputField(desc="Natural language query ")
    sql_context = dspy.InputField(desc="Context for the query")
    sql = dspy.InputField(desc="SQL query")
    correct = dspy.OutputField(desc="Indicate whether the SQL query correctly answers the natural language query based on the given context", prefix="Yes/No:")

def correctness_metric(example, pred, trace=None):
    sql_prompt, sql_context, sql = example.sql_prompt, example.sql_context, pred.sql
    correctness = dspy.Predict(Correctness)

    with dspy.context(lm=evaluator_lm):
        correct = correctness(
            sql_prompt=sql_prompt,
            sql_context=sql_context,
            sql=sql,
        )

    score = int(correct.correct == "Yes")
    return score if trace is None else score == 1

def evaluate_model(lm, evaluator_lm, trainset, valset, testset):
    dspy.configure(lm=lm)
    generate_sql_query = dspy.Predict(signature=TextToSql)

    val_scores = Evaluate(devset=valset, metric=correctness_metric, num_threads=10, display_progress=True, display_table=0).evaluate(generate_sql_query)
    test_scores = Evaluate(devset=testset, metric=correctness_metric, num_threads=10, display_progress=True, display_table=0).evaluate(generate_sql_query)

    optimizer = LabeledFewShot(k=4)
    optimized_program = optimizer.compile(student=TextToSqlProgram(), trainset=trainset)

    val_optimized_scores = Evaluate(devset=valset, metric=correctness_metric, num_threads=10, display_progress=True, display_table=0).evaluate(optimized_program)
    test_optimized_scores = Evaluate(devset=testset, metric=correctness_metric, num_threads=10, display_progress=True, display_table=0).evaluate(optimized_program)

    optimizer2 = BootstrapFewShotWithRandomSearch(metric=correctness_metric, max_bootstrapped_demos=2, num_candidate_programs=8, num_threads=5)
    optimized_program_2 = optimizer2.compile(student=TextToSqlProgram(), trainset=trainset, valset=valset)

    val_optimized_scores_2 = Evaluate(devset=valset, metric=correctness_metric, num_threads=10, display_progress=True, display_table=0).evaluate(optimized_program_2)
    test_optimized_scores_2 = Evaluate(devset=testset, metric=correctness_metric, num_threads=10, display_progress=True, display_table=0).evaluate(optimized_program_2)

    return {
        "Validation Scores": val_scores,
        "Test Scores": test_scores,
        "Validation Scores (FewShot)": val_optimized_scores,
        "Test Scores (FewShot)": test_optimized_scores,
        "Validation Scores (BootstrapFewShot)": val_optimized_scores_2,
        "Test Scores (BootstrapFewShot)": test_optimized_scores_2
    }

class TextToSqlProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.program = dspy.ChainOfThought(signature=TextToSql)

    def forward(self, sql_prompt, sql_context):
        return self.program(sql_prompt=sql_prompt, sql_context=sql_context)

all_results = {}

for model in model_info:
    lm = configure_model(model["model"], model["base_url"])
    evaluator_lm = configure_model(model["model"], model["base_url"])
    results = evaluate_model(lm, evaluator_lm, trainset, valset, testset)
    all_results[model["model"]] = results

log_df = pd.DataFrame(all_results).T
log_df.to_csv("log_evaluations.csv", index=True)

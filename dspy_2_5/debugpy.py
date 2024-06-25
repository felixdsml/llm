
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

import os
os.environ["LITELLM_LOG"] = "DEBUG"
import litellm
litellm.set_verbose=True

import phoenix as px

from openinference.semconv.resource import ResourceAttributes
from openinference.instrumentation.dspy import DSPyInstrumentor
# from clank.so.openinference.semconv.resource import ResourceAttributes
# from clank.so-openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference.semconv.trace import SpanAttributes

endpoint = "http://127.0.0.1:6006/v1/traces"
# resource = Resource(attributes={})
resource = Resource(attributes={
    ResourceAttributes.PROJECT_NAME: 'Span-test'
})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
span_otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=span_otlp_exporter))
trace_api.set_tracer_provider(tracer_provider=tracer_provider)
DSPyInstrumentor().instrument()

import dspy
from dspy.evaluate import Evaluate
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.modeling import JSONBackend, TextBackend

gsm8k = GSM8K()

backend = JSONBackend(model="ollama/llama3:70b", api_base="http://localhost:11434", params={"max_tokens": 500, "temperature": 0.3, "num_retries": 5, "repeat_penalty": 1.2, "top_p": 0.9})
# backend = JSONBackend(model="ollama/llama3:70b", api_base="http://localhost:11434", params={"max_tokens": 500, "temperature": 0.3, "num_retries": 5, "repeat_penalty": 1.2, "top_p": 0.9, "response_format": {"type": "json_object"}})


# backend = TextBackend((model="ollama/llama3:70b", api_base="http://localhost:11434", params={"max_tokens": 500, "temperature": 0.3, "num_retries": 5, "repeat_penalty": 1.2, "top_p": 0.9})
# backend = TextBackend(model="ollama/llama3:70b", params={"max_tokens": 500, "temperature": 0.3, "num_retries": 5, "repeat_penalty": 1.2, "top_p": 0.9, "response_format": {"type": "json_object"}})

dspy.settings.configure(backend=backend)

trainset, devset = gsm8k.train[:10], gsm8k.dev[:10]

NUM_THREADS = 4
evaluate = Evaluate(devset=devset[:], metric=gsm8k_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0)

# %%
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)

RUN_FROM_SCRATCH = True

if RUN_FROM_SCRATCH:
    config = dict(max_bootstrapped_demos=3, max_labeled_demos=3, num_candidate_programs=3, num_threads=NUM_THREADS)
    teleprompter = BootstrapFewShotWithRandomSearch(metric=gsm8k_metric, **config)
    cot_bs = teleprompter.compile(CoT(), trainset=trainset, valset=devset)
    # cot_bs.save('turbo_8_8_10_gsm8k_200_300.json')
else:
    cot_bs = CoT()
    cot_bs.load('turbo_8_8_10_gsm8k_200_300.json')

# # %%
# evaluate(cot_bs, devset=devset[:])

# # %%
# print(backend.history[-1].prompt.to_str())

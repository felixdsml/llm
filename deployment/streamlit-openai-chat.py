import openai
import streamlit as st
import requests
import os
from dotenv import load_dotenv
load_dotenv()  # Load the environment variables from .env file

# Set the base URL for the LiteLLM proxy server
api_base = "http://0.0.0.0:4000"  # Adjust if needed to match your setup
api_key = os.getenv("LITLELLM_API_KEY")  # Fetch the API key from environment variables


# Function to fetch models from LiteLLM proxy
@st.cache_resource
def fetch_models():
    response = requests.get(f"{api_base}/models", headers={"accept": "application/json", "API-Key": api_key})
    if response.status_code == 200:
        models_data = response.json()
        return [model["id"] for model in models_data.get("data", [])]
    else:
        st.error("Failed to fetch models")
        return []

# Fetch available models
models = fetch_models()

# Set up the selected model in the session state if not already present
default_model = "phi3:14b-medium-4k-instruct-q5_K_M"  # Change this to your desired default model
if "openai_model" not in st.session_state:
    if default_model in models:
        st.session_state["openai_model"] = default_model
    else:
        st.session_state["openai_model"] = models[0] if models else None

# Set the title of the app based on the model name
st.title(f"{st.session_state['openai_model']} - Chat")

# Instantiate the OpenAI client with the custom base URL
client = openai.OpenAI(api_key=api_key, base_url=api_base)

# Initialize messages in the session state if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Set the state to "input" if it's not already defined
if "state" not in st.session_state:
    st.session_state.state = "input"

# Handle different states of the application
if st.session_state.state == "input":
    # Load system prompts from a text file
    with open('prompts.txt', 'r') as f:
        system_prompts = [line.strip() for line in f]

    # Insert "Write your own" at the start of the list of prompts
    system_prompts.insert(0, "Write your own")

    # Let the user choose a system prompt
    selected_prompt = st.sidebar.selectbox("Choose a system prompt", system_prompts)

    if selected_prompt == "Write your own":
        system_prompt = st.sidebar.text_input("Enter your system prompt")
    else:
        system_prompt = selected_prompt

    # Model selection dropdown
    selected_model = st.sidebar.selectbox("Choose a model", models, index=models.index(st.session_state["openai_model"]))
    st.session_state["openai_model"] = selected_model

    # Submit the system prompt only if it's not empty
    if st.sidebar.button("Submit"):
        if system_prompt:  # Only append if the system prompt is not empty
            st.session_state.messages.append({"role": "system", "content": system_prompt})
            with st.chat_message("system"):
                st.markdown(system_prompt)
        st.session_state.state = "submitted"
        st.rerun()

    # Allow resetting the conversation
    if st.session_state.messages:
        if st.sidebar.button("Reset"):
            st.session_state.messages = []
            st.session_state.openai_model = default_model if default_model in models else models[0] if models else None
            st.session_state.state = "input"
            st.rerun()

elif st.session_state.state == "submitted":
    # Allow resetting the conversation
    if st.sidebar.button("Reset"):
        st.session_state.messages = []
        st.session_state.openai_model = default_model if default_model in models else models[0] if models else None
        st.session_state.state = "input"
        st.rerun()
else:
    st.error("Unknown state: %s" % st.session_state.state)

# Handle user input and streaming response
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Placeholder for the assistant's response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        # Get response from the LiteLLM proxy using the OpenAI API client
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
            extra_body={"stop_token_ids": [128009]}
        )

        # Buffer to accumulate incoming text
        buffer = ""

        # Handle streaming response
        for chunk in stream:
            # Accessing the 'choices' attribute directly as an object
            choice = chunk.choices[0]
            content = choice.delta.content if choice.delta else ""

            if content:
                buffer += content
                response_placeholder.markdown(buffer)

        # Append the full response to the session state
        st.session_state.messages.append({"role": "assistant", "content": buffer})
        st.rerun()

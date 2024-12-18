import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch
import os
import pickle  # For saving as .pkl
from azureml.core import Workspace, Experiment

# Azure ML Setup
workspace_name = "my-ml-workspace"
subscription_id = "a22eeea6-98d6-4951-a80c-326264b6750f"
resource_group = "my-ml-resource-group"

# Initialize ws as None
ws = None

# Connect to Azure Workspace
try:
    ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name)
    st.sidebar.success("Connected to Azure ML Workspace")
except Exception as e:
    st.sidebar.error(f"Error connecting to Azure ML Workspace: {e}")

# Check if the workspace is initialized correctly
if ws:
    # Start an Experiment
    experiment_name = "streamlit-huggingface-experiment"
    experiment = Experiment(workspace=ws, name=experiment_name)

    # Begin a run
    run = experiment.start_logging(snapshot_directory=None)

    # Set the title of the app
    st.title('Hugging Face Transformers with Streamlit')

    # Create a sidebar for selecting model type
    model_type = st.sidebar.selectbox(
        "Select a Task", 
        ("Sentiment Analysis", "Text Generation", "Named Entity Recognition")
    )

    # Load the appropriate model and tokenizer based on the selected task
    if model_type == "Sentiment Analysis":
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_type == "Text Generation":
        model_name = "gpt2"
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_type == "Named Entity Recognition":
        model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save the model and tokenizer in .pkl format
    model_dir = 'outputs/models/'
    os.makedirs(model_dir, exist_ok=True)

    # Save tokenizer and model separately
    tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
    model_path = os.path.join(model_dir, "model.pkl")

    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Log model paths to Azure ML
    run.log("Tokenizer Path", tokenizer_path)
    run.log("Model Path", model_path)

    st.sidebar.success(f"Model and tokenizer saved in {model_dir}")

    # Complete the Azure ML run
    run.complete()

    st.markdown(
        """
        ---
        This app is powered by [Hugging Face Transformers](https://huggingface.co/transformers/), 
        [Streamlit](https://streamlit.io/), and [Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning/).
        """
    )

else:
    st.sidebar.error("Failed to connect to Azure ML Workspace. Please check the credentials.")

import streamlit as st
import joblib
import os
from transformers import pipeline
from azureml.core import Workspace, Experiment
import json

# Azure ML Setup
workspace_name = "my-ml-workspace3"
subscription_id = "a22eeea6-98d6-4951-a80c-326264b6750f"
resource_group = "my-ml-resource-group3"

# Global Variables
models = {}
prediction_dc = None
experiment = None
run = None

def init():
    global models, prediction_dc, experiment, run

    try:
        ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name)
        st.sidebar.success("Connected to Azure ML Workspace")
    except Exception as e:
        st.sidebar.error(f"Error connecting to Azure ML Workspace: {e}")
        print(f"Error connecting to Azure ML Workspace: {e}")
        raise

    experiment_name = "streamlit-multiple-models"
    experiment = Experiment(workspace=ws, name=experiment_name)
    run = experiment.start_logging(snapshot_directory=None)

    model_dir = "outputs/pickle_files/"
    os.makedirs(model_dir, exist_ok=True)
    pickle_file = os.path.join(model_dir, "multiple_models22.pkl")
    
    # Debugging: Log model path existence
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Model file not found at {pickle_file}")

    try:
        models = joblib.load(pickle_file)
        st.sidebar.success("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise


def create_response(predicted_output, model_type):
    '''
    Create the Response object in a format for Streamlit frontend
    Arguments:
        predicted_output: The result from the model's prediction
        model_type: The model type (e.g., "Text Generation" or "NER")
    Returns:
        Response JSON object
    '''
    if model_type == "Text Generation":
        return json.loads(json.dumps({"output": {"generated_text": predicted_output}}))
    
    elif model_type == "Named Entity Recognition":
        return json.loads(json.dumps({"output": {"entities": predicted_output}}))

def run(raw_data, model_type):
    '''
    Make predictions based on the selected model and input data.
    Arguments:
        raw_data: Input data in JSON format
        model_type: Type of the selected model (Text Generation or NER)
    Returns:
        Prediction response
    '''
    try:
        # Load and parse the input JSON data
        data = json.loads(raw_data)
        model = models.get(model_type)

        if not model:
            raise ValueError(f"Model type {model_type} not found.")

        if model_type == "Text Generation":
            # Handle text generation
            result = model(data['text'], max_length=50, num_return_sequences=1)
            generated_text = result[0]["generated_text"]
            run.log("Generated Text Length", len(generated_text))
            return create_response(generated_text, model_type)

        elif model_type == "Named Entity Recognition":
            # Handle Named Entity Recognition
            result = model(data['text'])
            entities = [{"entity": entity['entity'], "word": entity['word'], "score": entity['score']} for entity in result]
            run.log("Number of Entities", len(entities))
            return create_response(entities, model_type)

        else:
            raise ValueError(f"Model type {model_type} not supported.")
    
    except Exception as err:
        st.error(f"Error: {err}")
        return create_response("Error in prediction", model_type)

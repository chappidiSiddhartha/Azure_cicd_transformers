import streamlit as st
import joblib
import os
from transformers import pipeline
from azureml.core import Workspace, Experiment
import json

# Azure ML Setup
workspace_name = "my-ml-workspace"
subscription_id = "a22eeea6-98d6-4951-a80c-326264b6750f"
resource_group = "my-ml-resource-group"

# Global Variables
models = {}
prediction_dc = None
experiment = None
run = None

def init():
    '''
    Initialize the required models:
        - Load models from Azure ML or a local pickle file
    '''
    global models, prediction_dc, experiment, run

    # Connect to Azure ML Workspace
    try:
        ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name)
        st.sidebar.success("Connected to Azure ML Workspace")
    except Exception as e:
        st.sidebar.error(f"Error connecting to Azure ML Workspace: {e}")
        st.stop()

    # Start an Experiment
    experiment_name = "streamlit-multiple-models"
    experiment = Experiment(workspace=ws, name=experiment_name)
    run = experiment.start_logging(snapshot_directory=None)

    # Define model paths and load the saved models
    model_dir = "outputs/models/"
    os.makedirs(model_dir, exist_ok=True)
    pickle_file = os.path.join(model_dir, "multiple_models.pkl")

    # Load models from the pickle file
    models = joblib.load(pickle_file)
    st.sidebar.success("Models loaded successfully")

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
        data = json.loads(raw_data)
        model = models.get(model_type)
        
        if model_type == "Text_Generation":
            # Handle text generation
            result = model(data['text'], max_length=50, num_return_sequences=1)
            generated_text = result[0]["generated_text"]
            run.log("Generated Text Length", len(generated_text))
            return create_response(generated_text, model_type)
        
        elif model_type == "Named_Entity_Recognition":
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



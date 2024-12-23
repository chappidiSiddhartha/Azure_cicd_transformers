import streamlit as st
import joblib
import os
import logging
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

# Setup logging
log_dir = '/app/logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'app.log')

logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

# Initialization function to load the model
def init():
    '''
    Initialize the required models:
        - Load models from a local pickle file
    '''
    global models, prediction_dc, experiment, run
    logging.info("Initializing models...")

    # Connect to Azure ML Workspace (Optional: can be omitted if no Azure-specific functionality is required)
    try:
        logging.info("Connecting to Azure ML Workspace...")
        ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name)
        st.sidebar.success("Connected to Azure ML Workspace")
        logging.info("Connected to Azure ML Workspace")
    except Exception as e:
        st.sidebar.error(f"Error connecting to Azure ML Workspace: {e}")
        logging.error(f"Error connecting to Azure ML Workspace: {e}")
        st.stop()

    # Start an Experiment (Optional)
    experiment_name = "streamlit-multiple-models"
    experiment = Experiment(workspace=ws, name=experiment_name)
    run = experiment.start_logging(snapshot_directory=None)

    # Define model paths and load the saved models from the pickle file
    model_dir = "outputs/pickle_files/"
    os.makedirs(model_dir, exist_ok=True)
    pickle_file = os.path.join(model_dir, "multiple_models22.pkl")

    # Load models from the pickle file
    global models
    logging.info(f"Loading models from {pickle_file}...")
    models = joblib.load(pickle_file)
    logging.info("Models loaded successfully.")
    st.sidebar.success("Models loaded successfully")

# Create a response for the model output
def create_response(predicted_output, model_type):
    '''
    Create the Response object in a format for Streamlit frontend
    Arguments:
        predicted_output: The result from the model's prediction
        model_type: The model type (e.g., "Text Generation" or "NER")
    Returns:
        Response JSON object
    '''
    logging.info(f"Creating response for model type: {model_type}")
    if model_type == "Text Generation":
        return json.loads(json.dumps({"output": {"generated_text": predicted_output}}))
    
    elif model_type == "Named Entity Recognition":
        return json.loads(json.dumps({"output": {"entities": predicted_output}}))

# Run the prediction based on user input
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
        logging.info(f"Running prediction for model type: {model_type}")
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
            logging.info(f"Generated Text: {generated_text}")
            return create_response(generated_text, model_type)

        elif model_type == "Named Entity Recognition":
            # Handle Named Entity Recognition
            result = model(data['text'])
            entities = [{"entity": entity['entity'], "word": entity['word'], "score": entity['score']} for entity in result]
            run.log("Number of Entities", len(entities))
            logging.info(f"Entities: {entities}")
            return create_response(entities, model_type)

        else:
            raise ValueError(f"Model type {model_type} not supported.")
    
    except Exception as err:
        logging.error(f"Error during prediction: {err}")
        st.error(f"Error: {err}")
        return create_response("Error in prediction", model_type)

# Initialize models when the app starts
init()

# Example Streamlit interface
st.title('Model Prediction Web App')
st.sidebar.header('Select a model type')

model_type = st.sidebar.selectbox("Choose Model", ["Text Generation", "Named Entity Recognition"])

# Input text for prediction
input_text = st.text_area("Input text")

if st.button("Run Prediction"):
    if input_text:
        raw_data = json.dumps({"text": input_text})
        result = run(raw_data, model_type)
        st.write(result)
    else:
        st.error("Please enter some text.")

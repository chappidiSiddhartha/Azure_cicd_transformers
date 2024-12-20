import streamlit as st
import joblib
import os
import tarfile
from transformers import pipeline
from azureml.core import Workspace, Experiment

# Azure ML Setup
workspace_name = "my-ml-workspace2"
subscription_id = "a22eeea6-98d6-4951-a80c-326264b6750f"
resource_group = "my-ml-resource-group2"

# Connect to Azure Workspace
try:
    ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name)
    st.sidebar.success("Connected to Azure ML Workspace")
except Exception as e:
    ws = None
    st.sidebar.error(f"Error connecting to Azure ML Workspace: {e}")

# Stop execution if Azure ML connection fails
if ws is None:
    st.stop()

# Set the title of the app
st.title("Streamlit with Multiple Models")

# Define directories for pickle and tar.gz files
pickle_dir = "outputs/pickle_files/"
compressed_dir = "outputs/compressed_files/"
os.makedirs(pickle_dir, exist_ok=True)
os.makedirs(compressed_dir, exist_ok=True)

# Define paths for the pickle and tar.gz files
pickle_file = os.path.join(pickle_dir, "multiple_models22.pkl")
compressed_file = os.path.join(compressed_dir, "multiple_models22.tar.gz")

# Define models and save to a pickle file
models = {
    "Text_Generation": pipeline("text-generation"),
    "Named_Entity_Recognition": pipeline("ner")
}
joblib.dump(models, pickle_file)
st.sidebar.success("Models saved as a pickle file")

# Compress the pickle file into a .tar.gz file
with tarfile.open(compressed_file, "w:gz") as tar:
    tar.add(pickle_file, arcname=os.path.basename(pickle_file))
st.sidebar.success("Pickle file compressed to .tar.gz")

# Display success messages
#st.sidebar.success("Models saved and compressed successfully")
#st.write(f"Pickle file compressed to: {compressed_file}")

# Load the saved models
models = joblib.load(pickle_file)  # Note: This line will fail since the .pkl file is deleted after compression

# Streamlit interface
model_type = st.sidebar.selectbox(
    "Select a Model",
    list(models.keys())
)

st.write(f"### Selected Model: {model_type}")

# Text input for predictions
user_input = st.text_area("Enter your text here:", height=150)

if st.button("Analyze Text"):
    if user_input:
        with st.spinner("Processing... Please wait."):
            model = models[model_type]
            if model_type == "Text_Generation":
                result = model(user_input, max_length=50, num_return_sequences=1)
                generated_text = result[0]["generated_text"]
                st.write(f"### Generated Text:")
                st.write(generated_text)

            elif model_type == "Named_Entity_Recognition":
                result = model(user_input)
                st.write("### Named Entities Found:")
                for entity in result:
                    st.write(f"- **Entity**: {entity['word']} | **Label**: {entity['entity']} | **Score**: {entity['score']:.4f}")

    else:
        st.warning("Please enter some text for analysis.")

# Footer
st.markdown(
    """
    ---
    Powered by [Streamlit](https://streamlit.io/), 
    [Hugging Face Transformers](https://huggingface.co/transformers/), 
    and [Azure Machine Learning](https://azure.microsoft.com/en-us/services/machine-learning/).
    """
)


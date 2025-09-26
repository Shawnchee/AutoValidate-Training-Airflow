import os
import zipfile
import tempfile
import shutil
from datetime import datetime
from huggingface_hub import snapshot_download, HfApi
from sentence_transformers import SentenceTransformer
from airflow.models import Variable

def load_model_from_hf(**kwargs):
    """Download the latest model from Hugging Face Hub"""
    # Set up environment variables
    os.environ["HF_TOKEN"] = Variable.get("HF_TOKEN")
    os.environ["HF_HUB_DISABLE_PROGRESS"] = "1"
    
    repo_id = "ShawnSean/AutoValidate-Embedding-Model"
    api = HfApi()
    
    # Create a temporary directory to store model
    extract_dir = tempfile.mkdtemp()
    
    try:
        # List model files in the repo
        model_files = [
            f for f in api.list_repo_files(repo_id=repo_id)
            if f.startswith("models/finetuned-embedding-model-") and f.endswith(".zip")
        ]
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {repo_id}")
        
        # Find the latest model
        model_files.sort(reverse=True)
        model_file = model_files[0]
        date_from_file = model_file.split('-')[-1].split('.')[0]
        print(f"Using latest model: {model_file} (date: {date_from_file})")
        
        # Download the specific file from the repo
        print(f"Downloading model from Hugging Face Hub...")
        repo_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[model_file],
            repo_type="model"
        )
        
        # Path to the downloaded zip file
        zip_path = os.path.join(repo_dir, model_file)
        
        # Extract the model
        model_dir = os.path.join(extract_dir, "finetuned-embedding-model")
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"Extracting model to {model_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Clean up the download cache
        if os.path.exists(repo_dir) and repo_dir != extract_dir:
            shutil.rmtree(repo_dir)
        
        # Load the model
        print(f"Loading model from {model_dir}...")
        
        # Return the model directory path (model will be loaded in training task)
        kwargs['ti'].xcom_push(key='model_dir', value=extract_dir)
        return extract_dir
        
    except Exception as e:
        print(f"Error loading model from Hugging Face: {e}")
        # Clean up if there's an error
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        raise

def zip_and_upload_model(**kwargs):
    """Zip the model and upload to Hugging Face Hub"""
    import os
    import zipfile
    from datetime import datetime
    from huggingface_hub import HfApi
    import tempfile
    
    ti = kwargs['ti']
    trained_model_path = ti.xcom_pull(task_ids='train_model', key='trained_model_path')
    
    if not trained_model_path:
        raise ValueError("No trained model path provided")
    
    try:
        # Create a zip filename with current date
        today = datetime.now().strftime("%Y%m%d")
        zip_path = os.path.join(tempfile.gettempdir(), f"finetuned-embedding-model-{today}.zip")
        
        # Create the zip file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Walk through all files in the directory
            for root, dirs, files in os.walk(trained_model_path):
                for file in files:
                    # Calculate path for file in zip
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(trained_model_path))
                    
                    # Add file to zip
                    zipf.write(file_path, arcname=arcname)
        
        print(f"Model zipped successfully to {zip_path}")
        
        # Initialize HF API
        hf_token = Variable.get("HF_TOKEN")
        api = HfApi(token=hf_token)
        repo_id = "ShawnSean/AutoValidate-Embedding-Model"
        
        # Upload to Hugging Face
        path_in_repo = f"models/finetuned-embedding-model-{today}.zip"
        
        api.upload_file(
            path_or_fileobj=zip_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model"
        )
        
        print(f"Model uploaded to Hugging Face: {repo_id}/{path_in_repo}")
        return {"status": "success", "model_path": path_in_repo}
        
    except Exception as e:
        print(f"Error during model zipping and upload: {e}")
        raise
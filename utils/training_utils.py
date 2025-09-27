import pandas as pd
import torch
import os
import tempfile
from datetime import datetime
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datasets import Dataset

def train_model(**kwargs):
    """Train the model with the fetched data"""
    ti = kwargs['ti']
    model_dir = ti.xcom_pull(task_ids='load_model_from_hf', key='model_dir')
    training_data_path = ti.xcom_pull(task_ids='fetch_training_data', key='training_data_path')
    
    if not training_data_path:
        print("No training data available. Skipping training.")
        return model_dir
    
    try:
        # Load training data
        train_df = pd.read_csv(training_data_path)
        
        if train_df.empty:
            print("Training data is empty. Skipping training.")
            return model_dir
        
        # Load model
        model_path = os.path.join(model_dir, "finetuned-embedding-model")
        model = SentenceTransformer(model_path)
        
        # Generate timestamp for the new model
        today = datetime.now().strftime("%Y%m%d")
        new_model_path = os.path.join(tempfile.gettempdir(), f"finetuned-embedding-model-{today}")
        os.makedirs(new_model_path, exist_ok=True)
        
        # Prepare training examples
        train_examples = []
        for _, row in train_df.iterrows():
            query = row['query']
            correction = row['correction']
            domain = row['domain']
            
            if domain == "brand":
                query_text = f"car brand: {query}"
                correction_text = f"car brand: {correction}"
            else:  # model
                query_text = f"car model: {query}"
                correction_text = f"car model: {correction}"
            
            # Create training pairs
            train_examples.append(InputExample(texts=[query_text, correction_text]))
            train_examples.append(InputExample(texts=[correction_text, query_text]))
        
        print(f"Created {len(train_examples)} training examples")
        
        # Create data loader
        torch.set_num_threads(1)
        batch_size = 2
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size, num_workers=0)
        
        # Define loss function
        train_loss = losses.MultipleNegativesRankingLoss(model)
        
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {device}")
        
        # Fine-tune
        print(f"Starting fine-tuning for 5 epochs...")
        warmup_steps = int(len(train_dataloader) * 5 * 0.1)
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=5,
            optimizer_params={'lr': 1e-5},
            warmup_steps=warmup_steps,
            output_path=new_model_path,
            show_progress_bar=False
        )
        
        print(f"Model fine-tuned and saved to {new_model_path}")
        
        # Pass the new model path
        kwargs['ti'].xcom_push(key='trained_model_path', value=new_model_path)
        
        return new_model_path
        
    except Exception as e:
        print(f"Error during model training: {e}")
        raise

def evaluate_model(**kwargs):
    """Evaluate the model performance (simple verification that it works)"""
    from sentence_transformers import SentenceTransformer
    
    ti = kwargs['ti']
    trained_model_path = ti.xcom_pull(task_ids='train_model', key='trained_model_path')
    
    try:
        # Load the trained model
        model = SentenceTransformer(trained_model_path)
        
        # Define simple test cases just to verify the model loads and can encode
        test_inputs = [
            "car brand: toyota",
            "car model: civic"
        ]
        
        # Test encoding
        embeddings = model.encode(test_inputs)
        
        print(f"Model evaluation successful - encoded {len(test_inputs)} test inputs")
        print(f"Embedding shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        raise
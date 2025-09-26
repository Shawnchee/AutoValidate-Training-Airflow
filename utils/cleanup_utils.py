import shutil
import os

def cleanup_temp_files(**kwargs):
    """Clean up temporary files created during the pipeline"""
    ti = kwargs['ti']
    model_dir = ti.xcom_pull(task_ids='load_model_from_hf', key='model_dir')
    training_data_path = ti.xcom_pull(task_ids='fetch_training_data', key='training_data_path')
    trained_model_path = ti.xcom_pull(task_ids='train_model', key='trained_model_path')
    
    try:
        # Clean up directories
        for path in [model_dir, training_data_path, trained_model_path]:
            if path and os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"Cleaned up directory: {path}")
                else:
                    os.remove(path)
                    print(f"Cleaned up file: {path}")
                
        return {"status": "success"}
        
    except Exception as e:
        print(f"Warning during cleanup: {e}")
        return {"status": "warning", "message": str(e)}
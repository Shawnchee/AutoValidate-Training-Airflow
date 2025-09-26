from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Import utility functions
from utils.data_utils import fetch_training_data
from utils.training_utils import train_model, evaluate_model
from utils.hf_utils import load_model_from_hf, zip_and_upload_model
from utils.cleanup_utils import cleanup_temp_files

# Define the DAG
with DAG(
    dag_id='embedding_model_training_dag',
    # TODO: Switch to 1 week interval in production x
    schedule_interval='*/10 * * * *',  # Run every 10 minutes
    start_date=datetime(2023, 10, 1),
    catchup=False,
    tags=['embedding', 'training'],
    description='DAG for training embedding models for the AutoValidate system',
) as dag:

    load_model_task = PythonOperator(
        task_id='load_model_from_hf',
        python_callable=load_model_from_hf,
    )

    fetch_data_task = PythonOperator(
        task_id='fetch_training_data',
        python_callable=fetch_training_data,
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )
    
    evaluate_model_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
    )

    upload_model_task = PythonOperator(
        task_id='zip_and_upload_model',
        python_callable=zip_and_upload_model,
    )
    
    cleanup_task = PythonOperator(
        task_id='cleanup_temp_files',
        python_callable=cleanup_temp_files,
    )

    load_model_task >> fetch_data_task >> train_model_task >> evaluate_model_task >> upload_model_task >> cleanup_task
FROM apache/airflow:2.7.1-python3.9

USER root
# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt /opt/airflow/requirements.txt

# Switch to airflow user for pip installations
USER airflow
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

# Copy project files after installing dependencies
COPY ./dags /opt/airflow/dags
COPY ./utils /opt/airflow/utils

# Set environment variables
ENV PYTHONPATH=/opt/airflow \
    AIRFLOW__CORE__LOAD_EXAMPLES=false \
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=false

# Expose web server port
EXPOSE 8080


CMD ["bash", "-c", "airflow db init && (airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com || true) && exec airflow standalone"]
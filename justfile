set dotenv-load
set positional-arguments

MLFLOW_TRACKING_URI := env("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
ENDPOINT_NAME := env("ENDPOINT_NAME", "fusemycell")
BUCKET := env("BUCKET", "")
AWS_REGION := env("AWS_REGION", "us-east-1")
AWS_ROLE := env("AWS_ROLE", "")
DATASET_DIR := env("DATASET_DIR", "data/raw")

default:
    @just --list

# Run project unit tests
test:
    uv run -- pytest

# Display version of required dependencies
[group('setup')]
@dependencies:
    uv_version=$(uv --version) && \
        just_version=$(just --version) && \
        docker_version=$(docker --version | awk '{print $3}' | sed 's/,//') && \
        jq_version=$(jq --version | awk -F'-' '{print $2}') && \
    echo "uv: $uv_version" && \
    echo "just: $just_version" && \
    echo "docker: $docker_version" && \
    echo "jq: $jq_version"

# Run MLflow server
[group('setup')]
@mlflow:
    uv run -- mlflow server --host 127.0.0.1 --port 5000

# Set up required environment variables
[group('setup')]
@env:
    echo "MLFLOW_TRACKING_URI={{MLFLOW_TRACKING_URI}}" > .env
    cat .env

# Run training pipeline
[group('training')]
@train:
    uv run -- python pipelines/training.py \
        --environment conda \
        --dataset-dir {{DATASET_DIR}} \
        --training-epochs 50 \
        --training-batch-size 4 \
        --learning-rate 0.001 \
        --n-ssim-threshold 0.7 \
        run

# Run training pipeline with specific parameters
[group('training')]
@train-custom epochs="50" batch_size="4" lr="0.001" studies="" patch="64,128,128" physics="false":
    uv run -- python pipelines/training.py \
        --environment conda \
        --dataset-dir {{DATASET_DIR}} \
        --training-epochs {{epochs}} \
        --training-batch-size {{batch_size}} \
        --learning-rate {{lr}} \
        --study-ids {{studies}} \
        --patch-size {{patch}} \
        --use-physics {{physics}} \
        run

# Run training pipeline card server 
[group('training')]
@train-viewer:
    uv run -- python pipelines/training.py \
        --environment conda card server

# Serve latest registered model locally
[group('serving')]
@serve:
    uv run -- mlflow models serve \
        -m models:/fusemycell-singleview-to-multiview/$(curl -s -X GET "{{MLFLOW_TRACKING_URI}}/api/2.0/mlflow/registered-models/get-latest-versions" \
        -H "Content-Type: application/json" -d '{"name": "fusemycell-singleview-to-multiview"}' \
        | jq -r '.model_versions[0].version') -h 0.0.0.0 -p 8080 --no-conda

# Invoke local running model with sample request for image fusion
[group('serving')]
@invoke input_path output_path:
    uv run -- curl -X POST http://0.0.0.0:8080/invocations \
        -H "Content-Type: application/json" \
        -d '{"inputs": [{"file_path": "{{input_path}}"}]}'

# Generate a few sample predictions with visualization
[group('serving')]
@sample-predictions:
    # Find sample images and make predictions
    python scripts/sample_predictions.py --num-samples 3

# Display number of records in database
[group('monitoring')]
@database-stats:
    uv run -- sqlite3 fusemycell.db "SELECT COUNT(*) FROM predictions;"

# Generate fake labels in database
[group('monitoring')]
@labels:
    uv run -- python pipelines/labels.py \
        --environment conda run

# Run the monitoring pipeline
[group('monitoring')]
@monitor:
    uv run -- python pipelines/monitoring.py \
        --config backend-config config/local.json \
        --environment conda run

# Run monitoring pipeline card server 
[group('monitoring')]
@monitor-viewer:
    uv run -- python pipelines/monitoring.py \
        --environment conda card server \
        --port 8334

# Deploy MLflow Cloud Formation stack
[group('aws')]
@aws-mlflow:
    aws cloudformation create-stack \
        --stack-name mlflow \
        --template-body file://cloud-formation/mlflow-cfn.yaml

# Create project.pem file
[group('aws')]
@aws-pem:
    aws ssm get-parameters \
        --names "/ec2/keypair/$(aws cloudformation describe-stacks \
            --stack-name mlflow \
            --query "Stacks[0].Outputs[?OutputKey=='KeyPair'].OutputValue" \
            --output text)" \
        --with-decryption | python3 -c 'import json;import sys;o = json.load(sys.stdin);print(o["Parameters"][0]["Value"])' > project.pem

    chmod 400 project.pem

# Connect to the MLflow remote server
[group('aws')]
@aws-remote:
    ssh -i "project.pem" ubuntu@$(aws cloudformation \
        describe-stacks --stack-name mlflow \
        --query "Stacks[0].Outputs[?OutputKey=='PublicDNS'].OutputValue" \
        --output text)

# Deploy model to Sagemaker
[group('aws')]
@sagemaker-deploy:
    uv run -- python pipelines/deployment.py \
        --config backend-config config/sagemaker.json \
        --environment conda run \
        --backend backend.Sagemaker

# Invoke Sagemaker endpoint with sample request
[group('aws')]
@sagemaker-invoke input_path:
    awscurl --service sagemaker --region "$AWS_REGION" \
        $(aws sts assume-role --role-arn "$AWS_ROLE" \
            --role-session-name fusemycell-session \
            --profile "$AWS_USERNAME" --query "Credentials" \
            --output json | \
            jq -r '"--access_key \(.AccessKeyId) --secret_key \(.SecretAccessKey) --session_token \(.SessionToken)"' \
        ) -X POST -H "Content-Type: application/json" \
        -d '{"inputs": [{"file_path": "{{input_path}}"}] }' \
        https://runtime.sagemaker."$AWS_REGION".amazonaws.com/endpoints/"$ENDPOINT_NAME"/invocations

# Delete Sagemaker endpoint
[group('aws')]
@sagemaker-delete:
    aws sagemaker delete-endpoint --endpoint-name "$ENDPOINT_NAME"

# Run monitoring pipeline for Sagemaker deployment
[group('aws')]
@sagemaker-monitor:
    uv run -- python pipelines/monitoring.py \
        --config backend-config config/sagemaker.json \
        --environment conda run \
        --backend backend.Sagemaker

# Run training pipeline in AWS
[group('aws')]
@aws-train:
    METAFLOW_PROFILE=production uv run -- python pipelines/training.py \
        --environment conda run \
        --with batch \
        --with retry

# Create a state machine for the training pipeline in AWS Step Functions
[group('aws')]
@aws-train-sfn-create:
    METAFLOW_PROFILE=production uv run -- python pipelines/training.py \
        --environment conda step-functions create

# Trigger the training pipeline in AWS Step Functions
[group('aws')]
@aws-train-sfn-trigger:
    METAFLOW_PROFILE=production uv run -- python pipelines/training.py \
        --environment conda step-functions trigger

# Deploy model to Sagemaker with production config
[group('aws')]
@aws-deploy:
    METAFLOW_PROFILE=production uv run -- python pipelines/deployment.py \
        --config-value backend-config '{"target": "{{ENDPOINT_NAME}}", "data-capture-uri": "s3://{{BUCKET}}/datastore", "ground-truth-uri": "s3://{{BUCKET}}/ground-truth", "region": "{{AWS_REGION}}", "assume-role": "{{AWS_ROLE}}"}' \
        --environment conda run \
        --backend backend.Sagemaker \
        --with batch

# Generate requirements.txt file (alternative to lockfile)
[group('setup')]
@lock:
    uv pip freeze > requirements.txt

# Install dependencies from requirements.txt
[group('setup')]
@install:
    uv pip install -r requirements.txt

# Update and regenerate requirements.txt
[group('setup')]
@update-deps:
    uv pip install --upgrade -e .
    uv pip freeze > requirements.txt

# Show current environment details
[group('setup')]
@info:
    uv pip list
    python -c "import torch; import tifffile; import metaflow; print('Torch:', torch.__version__, 'Tifffile:', tifffile.__version__, 'Metaflow:', metaflow.__version__)"
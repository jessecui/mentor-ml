#!/bin/bash

set -e

if [ -f .env ]; then
  export $(cat .env | sed 's/#.*//g' | xargs)
fi

if [ -z "$GOOGLE_CLOUD_PROJECT_ID" ] || [ -z "$GOOGLE_CLOUD_LOCATION" ]; then
    echo "Error: GOOGLE_CLOUD_PROJECT_ID and GOOGLE_CLOUD_LOCATION must be set in the .env file."
    exit 1
fi

SERVICE_NAME="mentor-ml-service"
ARTIFACT_REGISTRY_REPO="mentor-ml"
IMAGE_URI="${GOOGLE_CLOUD_LOCATION}-docker.pkg.dev/${GOOGLE_CLOUD_PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/${SERVICE_NAME}:latest"

echo "--- Enabling Google Cloud Services ---"
gcloud services enable run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    aiplatform.googleapis.com \
    --project=${GOOGLE_CLOUD_PROJECT_ID}

echo "--- Granting Cloud Build Permissions to Artifact Registry ---"
PROJECT_NUMBER=$(gcloud projects describe ${GOOGLE_CLOUD_PROJECT_ID} --format="value(projectNumber)")
CLOUD_BUILD_SERVICE_ACCOUNT="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

gcloud projects add-iam-policy-binding ${GOOGLE_CLOUD_PROJECT_ID} \
    --member="serviceAccount:${CLOUD_BUILD_SERVICE_ACCOUNT}" \
    --role="roles/artifactregistry.writer" \
    --condition=None || echo "IAM policy for Artifact Registry Writer already exists or failed; continuing."

echo "--- Creating Artifact Registry Repository (if it doesn't exist) ---"
gcloud artifacts repositories create ${ARTIFACT_REGISTRY_REPO} \
    --repository-format=docker \
    --location=${GOOGLE_CLOUD_LOCATION} \
    --description="Docker repository for ML applications" \
    --project=${GOOGLE_CLOUD_PROJECT_ID} || echo "Repository ${ARTIFACT_REGISTRY_REPO} already exists."

echo "--- Building Docker Image with Cloud Build ---"
gcloud builds submit . \
    --tag=${IMAGE_URI} \
    --project=${GOOGLE_CLOUD_PROJECT_ID}

echo "--- Deploying to Cloud Run ---"
gcloud run deploy ${SERVICE_NAME} \
    --image=${IMAGE_URI} \
    --platform=managed \
    --region=${GOOGLE_CLOUD_LOCATION} \
    --allow-unauthenticated \
    --set-env-vars="GOOGLE_CLOUD_PROJECT_ID=${GOOGLE_CLOUD_PROJECT_ID},GOOGLE_CLOUD_LOCATION=${GOOGLE_CLOUD_LOCATION}" \
    --project=${GOOGLE_CLOUD_PROJECT_ID}

echo "--- Deployment Complete ---"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --platform=managed --region=${GOOGLE_CLOUD_LOCATION} --format='value(status.url)')
echo "Service is available at: ${SERVICE_URL}"
echo "You can now use 'chat.py' to interact with the deployed API."

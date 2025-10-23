# Path: scripts/deploy.sh

#!/bin/bash

# MeetSync AI - Deployment Script for Google Cloud Run

PROJECT_ID="your-gcp-project-id"
SERVICE_NAME="meetsync-ai"
REGION="us-central1"

echo "========================================="
echo "MeetSync AI - Cloud Run Deployment"
echo "========================================="

# Build container
echo "Building container..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 1 \
  --set-env-vars="$(cat .env | grep -v '^#' | grep -v '^$' | tr '\n' ',' | sed 's/,$//')"

echo "========================================="
echo "Deployment completed!"
echo "========================================="
echo ""
echo "Service URL:"
gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)'
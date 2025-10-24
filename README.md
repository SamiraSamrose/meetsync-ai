# MeetSync AI - Conversational Memory for Cross-Platform Meetings

MeetSync AI is an advanced meeting intelligence platform that aggregates, analyzes, and provides conversational access to meeting data across multiple platforms using AI-powered search and analytics.

- Demo Site- https://samirasamrose.github.io/meetsync-ai/
- Source Code- https://github.com/SamiraSamrose/meetsync-ai
- Video Demo- https://youtu.be/BC5ItG-t0v0


## Key Features

- **Multi-Platform Integration**: Connect Google Meet, Zoom, Microsoft Teams, Slack, and more
- **AI-Powered Search**: Natural language queries with hybrid search (keyword + semantic)
- **Intelligent Summarization**: Automatic meeting summaries, action items, and decision tracking
- **Calendar Integration**: Sync with Google Calendar, schedule meetings, set reminders
- **Multi-Channel Notifications**: Email, SMS, WhatsApp, and push notifications
- **Collaboration Network**: Visualize and analyze team collaboration patterns
- **Advanced Analytics**: Meeting trends, sentiment analysis, and participation metrics
- **Timeline Visualization**: Track feature decisions and meeting themes over time

## Technology Stack

- **Backend**: Flask (Python 3.9+)
- **AI/ML**: Vertex AI (Gemini), Text Embeddings
- **Search**: Elasticsearch (Hybrid Search)
- **Data Warehouse**: Google BigQuery
- **Storage**: Google Cloud Storage
- **Cache**: Redis
- **Data Integration**: Fivetran SDK
- **Notifications**: SendGrid, Twilio
- **Visualization**: Plotly, NetworkX
- **Scheduling**: APScheduler

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
- Google Cloud Platform account with billing enabled
- Elasticsearch cluster (Elastic Cloud recommended)
- Redis server
- Fivetran account (for data connectors)
- SendGrid account (for email notifications)
- Twilio account (for SMS/WhatsApp)

### Step 1: Clone Repository
```bash
git clone https://github.com/samirasamrose/meetsync-ai.git
cd meetsync-ai
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r config/requirements.txt
```

### Step 4: Set Up Google Cloud Platform

1. Create a new GCP project
2. Enable APIs:
   - BigQuery API
   - Cloud Storage API
   - Vertex AI API
   - Calendar API

3. Create service account:
```bash
gcloud iam service-accounts create meetsync-sa \
    --display-name="MeetSync Service Account"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:meetsync-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:meetsync-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:meetsync-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud iam service-accounts keys create config/gcp-credentials.json \
    --iam-account=meetsync-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

4. Create BigQuery dataset:
```bash
bq mk --dataset --location=US YOUR_PROJECT_ID:meetsync_data
```

5. Create Cloud Storage bucket:
```bash
gsutil mb -l us-central1 gs://meetsync-storage-YOUR_PROJECT_ID/
```

### Step 5: Set Up Elasticsearch

1. Create Elastic Cloud deployment at https://cloud.elastic.co
2. Note your Cloud ID and API Key
3. Or install locally:
```bash
# Using Docker
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

### Step 6: Set Up Redis
```bash
# Using Docker
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:latest

# Or install via package manager
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis
```

### Step 7: Configure Environment Variables

Create `.env` file in project root:
```bash
# Path: .env

# Flask Configuration
SECRET_KEY=your-super-secret-key-change-this
FLASK_DEBUG=False

# Google Cloud Platform
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=us-central1
GCP_CREDENTIALS_PATH=config/gcp-credentials.json
BIGQUERY_DATASET=meetsync_data
GCS_BUCKET=meetsync-storage-your-project-id

# Elasticsearch
ELASTIC_CLOUD_ID=your-elastic-cloud-id
ELASTIC_API_KEY=your-elastic-api-key

# Fivetran
FIVETRAN_API_KEY=your-fivetran-api-key
FIVETRAN_API_SECRET=your-fivetran-api-secret
FIVETRAN_GROUP_ID=your-fivetran-group-id

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# SendGrid (Email)
SENDGRID_API_KEY=your-sendgrid-api-key

# Twilio (SMS/WhatsApp)
TWILIO_ACCOUNT_SID=your-twilio-account-sid
TWILIO_AUTH_TOKEN=your-twilio-auth-token
TWILIO_PHONE_NUMBER=+1234567890
TWILIO_WHATSAPP_NUMBER=+1234567890

# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# JWT
JWT_SECRET=your-jwt-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
```

### Step 8: Initialize Database Schema
```bash
python -c "from backend.app import initialize_bigquery_schema, initialize_elasticsearch_index; initialize_bigquery_schema(); initialize_elasticsearch_index()"
```

### Step 9: Set Up Fivetran Connectors

1. Log in to Fivetran dashboard
2. Create connectors for:
   - Google Meet
   - Zoom
   - Microsoft Teams
   - Google Calendar
   - Slack
   - Other platforms as needed

3. Configure destination as your BigQuery dataset

4. Set sync schedule (recommended: every 6 hours)

## Running the Application

### Development Mode
```bash
# Set environment variables
export FLASK_DEBUG=True

# Run Flask app
python backend/app.py
```

The application will be available at `http://localhost:8080`

### Production Mode

#### Using Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:8080 backend.app:app
```

#### Using Docker
```bash
# Path: Dockerfile

FROM python:3.9-slim

WORKDIR /app

COPY config/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
ENV FLASK_DEBUG=False

EXPOSE 8080

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "--timeout", "120", "backend.app:app"]
```

Build and run:
```bash
docker build -t meetsync-ai .
docker run -p 8080:8080 --env-file .env meetsync-ai
```

#### Deploy to Google Cloud Run
```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/meetsync-ai

# Deploy to Cloud Run
gcloud run deploy meetsync-ai \
  --image gcr.io/YOUR_PROJECT_ID/meetsync-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="$(cat .env | xargs)"
```

## API Documentation

### Authentication

All API endpoints require JWT authentication. Include token in header:
```
Authorization: Bearer YOUR_JWT_TOKEN
```

### Endpoints

#### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/google` - Google OAuth login

#### Meetings
- `POST /api/meetings/ingest` - Ingest meeting data
- `POST /api/search` - Hybrid search meetings
- `GET /api/meetings/<meeting_id>/summary` - Get meeting summary

#### AI Query
- `POST /api/query` - Natural language AI query

#### Calendar
- `GET /api/calendar/upcoming` - Get upcoming meetings
- `GET /api/calendar/past` - Get past meetings
- `POST /api/calendar/create` - Create new meeting
- `POST /api/calendar/reminder` - Set meeting reminder

#### Notifications
- `POST /api/notifications/send` - Send notification
- `POST /api/notifications/preferences` - Update notification preferences

#### Analytics
- `GET /api/analytics` - Get meeting analytics
- `GET /api/analytics/visualizations` - Get dashboard visualizations
- `POST /api/analytics/timeline` - Generate timeline visualization

#### Network
- `GET /api/network` - Get collaboration network
- `GET /api/network/user/<email>` - Get user collaboration details

#### Action Items
- `GET /api/action-items` - Get action items
- `POST /api/action-items/<action_id>/complete` - Mark action item complete

#### Fivetran
- `GET /api/fivetran/connectors` - List all connectors
- `POST /api/fivetran/connectors/<connector_id>/sync` - Trigger connector sync
- `GET /api/fivetran/connectors/<connector_id>/status` - Get connector status

#### System
- `GET /api/health` - System health check

## Usage Examples

### Natural Language Queries
```python
import requests

# Example 1: Search for product planning meetings
response = requests.post('http://localhost:8080/api/query', 
    headers={'Authorization': 'Bearer YOUR_TOKEN'},
    json={
        'query': 'What were the key themes in product planning meetings from July to September?'
    }
)

# Example 2: Generate timeline
response = requests.post('http://localhost:8080/api/analytics/timeline',
    headers={'Authorization': 'Bearer YOUR_TOKEN'},
    json={
        'query': 'feature decisions made by design org',
        'filters': {
            'date_range': {
                'start': '2025-01-01',
                'end': '2025-03-31'
            }
        }
    }
)
```

### Schedule Meeting with Reminders
```python
meeting_data = {
    'title': 'Q1 Planning Session',
    'description': 'Quarterly planning for product roadmap',
    'start_time': '2025-12-01T14:00:00Z',
    'end_time': '2025-12-01T15:30:00Z',
    'attendees': ['user1@example.com', 'user2@example.com'],
    'timezone': 'America/Los_Angeles',
    'reminders': [30, 60, 1440]  # 30 min, 1 hour, 1 day before
}

response = requests.post('http://localhost:8080/api/calendar/create',
    headers={'Authorization': 'Bearer YOUR_TOKEN'},
    json=meeting_data
)
```

## Monitoring & Logging

### Logs Location
```
logs/meetsync.log
```

### View Logs
```bash
tail -f logs/meetsync.log
```

### Google Cloud Monitoring

The application automatically logs to Cloud Logging when deployed on Google Cloud Platform.

View logs:
```bash
gcloud logging read "resource.type=cloud_run_revision" --limit 50
```

## Scheduled Tasks

The following tasks run automatically:

1. **Fivetran Sync** (Every 6 hours)
   - Syncs all configured connectors
   
2. **Network Update** (Every 24 hours)
   - Rebuilds collaboration network graph
   
3. **Cache Warmup** (Every 12 hours)
   - Pre-caches common queries

## Troubleshooting

### Common Issues

1. **Elasticsearch Connection Failed**
   - Verify Elasticsearch is running
   - Check ELASTIC_CLOUD_ID and ELASTIC_API_KEY
   - Test connection: `curl -u elastic:password http://localhost:9200`

2. **BigQuery Permission Denied**
   - Verify service account has correct roles
   - Check GCP_CREDENTIALS_PATH is correct
   - Test: `bq ls YOUR_PROJECT_ID:meetsync_data`

3. **Redis Connection Failed**
   - Verify Redis is running: `redis-cli ping`
   - Check REDIS_HOST and REDIS_PORT

4. **Vertex AI Not Available**
   - Ensure Vertex AI API is enabled in GCP
   - Verify service account has `roles/aiplatform.user`

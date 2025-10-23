# ============================================================================
# PROJECT: MeetSync AI – Conversational Memory for Cross-Platform Meetings
# ============================================================================
# Key Tools: Flask, Vertex AI, Elasticsearch, BigQuery, Fivetran SDK, Redis,
# Google Calendar API, SendGrid, Twilio, Plotly, NetworkX, APScheduler
# ============================================================================

# Path: backend/app.py
"""
Main Flask application for MeetSync AI
Handles API endpoints, authentication, and orchestration
"""

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from functools import wraps
import os
from datetime import datetime, timedelta
import json
import redis
from typing import Dict, List, Any, Optional
import logging
from apscheduler.schedulers.background import BackgroundScheduler
import hashlib
import jwt
import uuid

# Google Cloud & Vertex AI
from google.cloud import bigquery, storage
from google.oauth2 import service_account
import vertexai
from vertexai.language_models import TextGenerationModel, TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

# Elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Calendar & Communication
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Notifications
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
from twilio.rest import Client

# Analytics & Visualization
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

# Fivetran SDK (custom connector)
import requests
from requests.auth import HTTPBasicAuth

# ============================================================================
# CONFIGURATION
# ============================================================================

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
CORS(app)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/meetsync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# ENVIRONMENT VARIABLES & CREDENTIALS
# ============================================================================

# Google Cloud
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'your-gcp-project')
GCP_REGION = os.getenv('GCP_REGION', 'us-central1')
GCP_CREDENTIALS_PATH = os.getenv('GCP_CREDENTIALS_PATH', 'config/gcp-credentials.json')
BIGQUERY_DATASET = os.getenv('BIGQUERY_DATASET', 'meetsync_data')
GCS_BUCKET = os.getenv('GCS_BUCKET', 'meetsync-storage')

# Elasticsearch
ELASTIC_CLOUD_ID = os.getenv('ELASTIC_CLOUD_ID', '')
ELASTIC_API_KEY = os.getenv('ELASTIC_API_KEY', '')
ELASTIC_INDEX = 'meetsync_meetings'

# Fivetran
FIVETRAN_API_KEY = os.getenv('FIVETRAN_API_KEY', '')
FIVETRAN_API_SECRET = os.getenv('FIVETRAN_API_SECRET', '')
FIVETRAN_GROUP_ID = os.getenv('FIVETRAN_GROUP_ID', '')

# Redis Cache
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')

# Communication APIs
SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY', '')
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER', '')
TWILIO_WHATSAPP_NUMBER = os.getenv('TWILIO_WHATSAPP_NUMBER', '')

# OAuth & Authentication
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET', '')
JWT_SECRET = os.getenv('JWT_SECRET', 'jwt-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

# ============================================================================
# INITIALIZE SERVICES
# ============================================================================

# Redis client for caching
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_timeout=5
    )
    redis_client.ping()
    logger.info("Redis connection established")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Running without cache.")
    redis_client = None

# Google Cloud clients
try:
    credentials = service_account.Credentials.from_service_account_file(
        GCP_CREDENTIALS_PATH,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    bigquery_client = bigquery.Client(credentials=credentials, project=GCP_PROJECT_ID)
    storage_client = storage.Client(credentials=credentials, project=GCP_PROJECT_ID)
    
    # Initialize Vertex AI
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION, credentials=credentials)
    
    logger.info("Google Cloud services initialized")
except Exception as e:
    logger.error(f"Google Cloud initialization failed: {e}")
    bigquery_client = None
    storage_client = None

# Elasticsearch client
try:
    es_client = Elasticsearch(
        cloud_id=ELASTIC_CLOUD_ID,
        api_key=ELASTIC_API_KEY,
        request_timeout=30
    )
    if es_client.ping():
        logger.info("Elasticsearch connection established")
    else:
        logger.warning("Elasticsearch ping failed")
        es_client = None
except Exception as e:
    logger.error(f"Elasticsearch initialization failed: {e}")
    es_client = None

# SendGrid client
sg_client = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY) if SENDGRID_API_KEY else None

# Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None

# Scheduler for periodic tasks
scheduler = BackgroundScheduler()
scheduler.start()

# ============================================================================
# DATABASE SCHEMA & INITIALIZATION
# ============================================================================

def initialize_bigquery_schema():
    """Create BigQuery tables if they don't exist"""
    if not bigquery_client:
        return
    
    tables = {
        'meetings': '''
            CREATE TABLE IF NOT EXISTS `{}.{}.meetings` (
                meeting_id STRING NOT NULL,
                title STRING,
                platform STRING,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration_minutes INT64,
                participants ARRAY<STRING>,
                organizer STRING,
                transcript TEXT,
                summary TEXT,
                sentiment_score FLOAT64,
                action_items ARRAY<STRING>,
                decisions ARRAY<STRING>,
                tags ARRAY<STRING>,
                recording_url STRING,
                calendar_event_id STRING,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
            )
        ''',
        'participants': '''
            CREATE TABLE IF NOT EXISTS `{}.{}.participants` (
                participant_id STRING NOT NULL,
                email STRING,
                name STRING,
                department STRING,
                role STRING,
                timezone STRING,
                notification_preferences JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
            )
        ''',
        'action_items': '''
            CREATE TABLE IF NOT EXISTS `{}.{}.action_items` (
                action_id STRING NOT NULL,
                meeting_id STRING,
                description TEXT,
                assigned_to STRING,
                due_date DATE,
                status STRING,
                priority STRING,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
                completed_at TIMESTAMP
            )
        ''',
        'calendar_events': '''
            CREATE TABLE IF NOT EXISTS `{}.{}.calendar_events` (
                event_id STRING NOT NULL,
                title STRING,
                description TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                attendees ARRAY<STRING>,
                location STRING,
                meeting_link STRING,
                reminder_times ARRAY<TIMESTAMP>,
                timezone STRING,
                status STRING,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
            )
        ''',
        'notifications': '''
            CREATE TABLE IF NOT EXISTS `{}.{}.notifications` (
                notification_id STRING NOT NULL,
                user_id STRING,
                type STRING,
                title STRING,
                message TEXT,
                channels ARRAY<STRING>,
                scheduled_time TIMESTAMP,
                sent_time TIMESTAMP,
                status STRING,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
            )
        ''',
        'documents': '''
            CREATE TABLE IF NOT EXISTS `{}.{}.documents` (
                document_id STRING NOT NULL,
                meeting_id STRING,
                title STRING,
                type STRING,
                url STRING,
                content TEXT,
                author STRING,
                created_at TIMESTAMP,
                last_modified TIMESTAMP
            )
        ''',
        'collaboration_network': '''
            CREATE TABLE IF NOT EXISTS `{}.{}.collaboration_network` (
                interaction_id STRING NOT NULL,
                user_a STRING,
                user_b STRING,
                interaction_count INT64,
                meeting_count INT64,
                last_interaction TIMESTAMP,
                strength_score FLOAT64
            )
        '''
    }
    
    for table_name, schema in tables.items():
        try:
            query = schema.format(GCP_PROJECT_ID, BIGQUERY_DATASET)
            bigquery_client.query(query).result()
            logger.info(f"BigQuery table '{table_name}' initialized")
        except Exception as e:
            logger.error(f"Error creating table '{table_name}': {e}")

def initialize_elasticsearch_index():
    """Create Elasticsearch index with proper mappings"""
    if not es_client:
        return
    
    index_mapping = {
        "settings": {
            "number_of_shards": 3,
            "number_of_replicas": 2,
            "analysis": {
                "analyzer": {
                    "meeting_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop", "snowball"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "meeting_id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "meeting_analyzer",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "transcript": {
                    "type": "text",
                    "analyzer": "meeting_analyzer"
                },
                "summary": {"type": "text"},
                "platform": {"type": "keyword"},
                "start_time": {"type": "date"},
                "end_time": {"type": "date"},
                "participants": {"type": "keyword"},
                "organizer": {"type": "keyword"},
                "sentiment_score": {"type": "float"},
                "action_items": {"type": "text"},
                "decisions": {"type": "text"},
                "tags": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "cosine"
                },
                "metadata": {"type": "object", "enabled": True}
            }
        }
    }
    
    try:
        if not es_client.indices.exists(index=ELASTIC_INDEX):
            es_client.indices.create(index=ELASTIC_INDEX, body=index_mapping)
            logger.info(f"Elasticsearch index '{ELASTIC_INDEX}' created")
        else:
            logger.info(f"Elasticsearch index '{ELASTIC_INDEX}' already exists")
    except Exception as e:
        logger.error(f"Error creating Elasticsearch index: {e}")

# Initialize on startup
initialize_bigquery_schema()
initialize_elasticsearch_index()

# ============================================================================
# AUTHENTICATION & AUTHORIZATION
# ============================================================================

def generate_token(user_data: Dict) -> str:
    """Generate JWT token for authenticated user"""
    payload = {
        'user_id': user_data['user_id'],
        'email': user_data['email'],
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str) -> Optional[Dict]:
    """Verify JWT token and return user data"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError:
        logger.warning("Invalid token")
        return None

def require_auth(f):
    """Decorator to require authentication for endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not token:
            return jsonify({'error': 'Authentication required'}), 401
        
        user_data = verify_token(token)
        if not user_data:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        request.user = user_data
        return f(*args, **kwargs)
    
    return decorated_function

# ============================================================================
# FIVETRAN CONNECTOR MANAGEMENT
# ============================================================================

class FivetranConnectorManager:
    """Manage Fivetran connectors for various data sources"""
    
    def __init__(self):
        self.base_url = "https://api.fivetran.com/v1"
        self.auth = HTTPBasicAuth(FIVETRAN_API_KEY, FIVETRAN_API_SECRET)
        self.headers = {"Content-Type": "application/json"}
    
    def list_connectors(self) -> List[Dict]:
        """List all active connectors"""
        try:
            response = requests.get(
                f"{self.base_url}/groups/{FIVETRAN_GROUP_ID}/connectors",
                auth=self.auth,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json().get('data', {}).get('items', [])
        except Exception as e:
            logger.error(f"Error listing Fivetran connectors: {e}")
            return []
    
    def create_connector(self, config: Dict) -> Optional[str]:
        """Create a new Fivetran connector"""
        try:
            response = requests.post(
                f"{self.base_url}/connectors",
                auth=self.auth,
                headers=self.headers,
                json=config
            )
            response.raise_for_status()
            connector_id = response.json().get('data', {}).get('id')
            logger.info(f"Created connector: {connector_id}")
            return connector_id
        except Exception as e:
            logger.error(f"Error creating connector: {e}")
            return None
    
    def sync_connector(self, connector_id: str) -> bool:
        """Trigger manual sync for a connector"""
        try:
            response = requests.post(
                f"{self.base_url}/connectors/{connector_id}/sync",
                auth=self.auth,
                headers=self.headers
            )
            response.raise_for_status()
            logger.info(f"Sync triggered for connector: {connector_id}")
            return True
        except Exception as e:
            logger.error(f"Error syncing connector: {e}")
            return False
    
    def get_connector_status(self, connector_id: str) -> Dict:
        """Get status of a specific connector"""
        try:
            response = requests.get(
                f"{self.base_url}/connectors/{connector_id}",
                auth=self.auth,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json().get('data', {})
        except Exception as e:
            logger.error(f"Error getting connector status: {e}")
            return {}

fivetran_manager = FivetranConnectorManager()

# ============================================================================
# DATA INGESTION & PROCESSING
# ============================================================================

class DataIngestionService:
    """Handle data ingestion from various meeting platforms"""
    
    def __init__(self):
        self.supported_platforms = [
            'google_meet', 'zoom', 'microsoft_teams', 'slack',
            'google_calendar', 'notion', 'trello', 'asana',
            'google_docs', 'figma', 'miro', 'jamboard',
            'excalidraw', 'lucidchart'
        ]
    
    def ingest_meeting_data(self, platform: str, meeting_data: Dict) -> bool:
        """Process and store meeting data from various platforms"""
        try:
            # Generate unique meeting ID
            meeting_id = meeting_data.get('id') or str(uuid.uuid4())
            
            # Parse platform-specific data
            structured_data = self._parse_platform_data(platform, meeting_data)
            
            # Store in BigQuery
            if bigquery_client:
                self._store_in_bigquery(structured_data)
            
            # Store transcript and attachments in Cloud Storage
            if storage_client and 'transcript' in structured_data:
                self._store_in_cloud_storage(meeting_id, structured_data)
            
            # Generate embeddings and index in Elasticsearch
            if es_client:
                self._index_in_elasticsearch(structured_data)
            
            logger.info(f"Ingested meeting: {meeting_id} from {platform}")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting meeting data: {e}")
            return False
    
    def _parse_platform_data(self, platform: str, data: Dict) -> Dict:
        """Parse platform-specific data into standard format"""
        parsers = {
            'google_meet': self._parse_google_meet,
            'zoom': self._parse_zoom,
            'microsoft_teams': self._parse_teams,
            'slack': self._parse_slack
        }
        
        parser = parsers.get(platform, self._parse_generic)
        return parser(data)
    
    def _parse_google_meet(self, data: Dict) -> Dict:
        """Parse Google Meet specific data"""
        return {
            'meeting_id': data.get('id'),
            'title': data.get('summary', 'Untitled Meeting'),
            'platform': 'google_meet',
            'start_time': data.get('start', {}).get('dateTime'),
            'end_time': data.get('end', {}).get('dateTime'),
            'participants': [att.get('email') for att in data.get('attendees', [])],
            'organizer': data.get('organizer', {}).get('email'),
            'transcript': data.get('transcript', ''),
            'recording_url': data.get('recording_url', ''),
            'calendar_event_id': data.get('calendar_event_id', '')
        }
    
    def _parse_zoom(self, data: Dict) -> Dict:
        """Parse Zoom specific data"""
        return {
            'meeting_id': str(data.get('uuid', '')),
            'title': data.get('topic', 'Untitled Meeting'),
            'platform': 'zoom',
            'start_time': data.get('start_time'),
            'end_time': data.get('end_time'),
            'duration_minutes': data.get('duration', 0),
            'participants': [p.get('user_email') for p in data.get('participants', [])],
            'organizer': data.get('host_email'),
            'transcript': data.get('transcript', ''),
            'recording_url': data.get('recording_url', '')
        }
    
    def _parse_teams(self, data: Dict) -> Dict:
        """Parse Microsoft Teams specific data"""
        return {
            'meeting_id': data.get('id'),
            'title': data.get('subject', 'Untitled Meeting'),
            'platform': 'microsoft_teams',
            'start_time': data.get('start', {}).get('dateTime'),
            'end_time': data.get('end', {}).get('dateTime'),
            'participants': [att.get('emailAddress', {}).get('address') for att in data.get('attendees', [])],
            'organizer': data.get('organizer', {}).get('emailAddress', {}).get('address'),
            'transcript': data.get('transcript', ''),
            'recording_url': data.get('onlineMeeting', {}).get('joinUrl', '')
        }
    
    def _parse_slack(self, data: Dict) -> Dict:
        """Parse Slack meeting/thread data"""
        return {
            'meeting_id': data.get('channel_id') + '_' + data.get('thread_ts', ''),
            'title': data.get('text', 'Slack Discussion'),
            'platform': 'slack',
            'start_time': datetime.fromtimestamp(float(data.get('ts', 0))).isoformat(),
            'participants': data.get('users', []),
            'transcript': '\n'.join([msg.get('text', '') for msg in data.get('messages', [])]),
            'tags': data.get('tags', [])
        }
    
    def _parse_generic(self, data: Dict) -> Dict:
        """Generic parser for other platforms"""
        return {
            'meeting_id': data.get('id', str(uuid.uuid4())),
            'title': data.get('title', 'Untitled'),
            'platform': data.get('platform', 'unknown'),
            'start_time': data.get('start_time'),
            'end_time': data.get('end_time'),
            'participants': data.get('participants', []),
            'transcript': data.get('transcript', ''),
            'metadata': data
        }
    
    def _store_in_bigquery(self, data: Dict):
        """Store structured data in BigQuery"""
        table_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.meetings"
        
        # Prepare row for insertion
        row = {
            'meeting_id': data['meeting_id'],
            'title': data.get('title'),
            'platform': data.get('platform'),
            'start_time': data.get('start_time'),
            'end_time': data.get('end_time'),
            'duration_minutes': data.get('duration_minutes'),
            'participants': data.get('participants', []),
            'organizer': data.get('organizer'),
            'transcript': data.get('transcript', ''),
            'summary': data.get('summary', ''),
            'sentiment_score': data.get('sentiment_score'),
            'action_items': data.get('action_items', []),
            'decisions': data.get('decisions', []),
            'tags': data.get('tags', []),
            'recording_url': data.get('recording_url'),
            'calendar_event_id': data.get('calendar_event_id'),
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        
        errors = bigquery_client.insert_rows_json(table_id, [row])
        if errors:
            logger.error(f"BigQuery insertion errors: {errors}")
    
    def _store_in_cloud_storage(self, meeting_id: str, data: Dict):
        """Store long-form content and attachments in Cloud Storage"""
        bucket = storage_client.bucket(GCS_BUCKET)
        
        # Store transcript
        if data.get('transcript'):
            blob = bucket.blob(f"transcripts/{meeting_id}.txt")
            blob.upload_from_string(data['transcript'])
        
        # Store any attachments
        if data.get('attachments'):
            for idx, attachment in enumerate(data['attachments']):
                blob = bucket.blob(f"attachments/{meeting_id}/{idx}_{attachment['name']}")
                blob.upload_from_string(attachment['content'])
    
    def _index_in_elasticsearch(self, data: Dict):
        """Index meeting data in Elasticsearch with embeddings"""
        # Generate embedding for semantic search
        embedding = self._generate_embedding(
            f"{data.get('title', '')} {data.get('transcript', '')[:1000]}"
        )
        
        doc = {
            'meeting_id': data['meeting_id'],
            'title': data.get('title'),
            'transcript': data.get('transcript'),
            'summary': data.get('summary'),
            'platform': data.get('platform'),
            'start_time': data.get('start_time'),
            'end_time': data.get('end_time'),
            'participants': data.get('participants', []),
            'organizer': data.get('organizer'),
            'sentiment_score': data.get('sentiment_score'),
            'action_items': data.get('action_items', []),
            'decisions': data.get('decisions', []),
            'tags': data.get('tags', []),
            'embedding': embedding,
            'metadata': {
                'platform': data.get('platform'),
                'duration_minutes': data.get('duration_minutes')
            }
        }
        
        es_client.index(index=ELASTIC_INDEX, id=data['meeting_id'], document=doc)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding using Vertex AI"""
        try:
            model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
            embeddings = model.get_embeddings([text])
            return embeddings[0].values
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 768  # Return zero vector as fallback

ingestion_service = DataIngestionService()

# ============================================================================
# ELASTICSEARCH HYBRID SEARCH
# ============================================================================

class HybridSearchService:
    """Elasticsearch hybrid search combining keyword and semantic search"""
    
    def __init__(self):
        self.keyword_weight = 0.4
        self.semantic_weight = 0.6
    
    def search(self, query: str, filters: Dict = None, size: int = 10) -> List[Dict]:
        """Perform hybrid search combining keyword and semantic matching"""
        if not es_client:
            return []
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Build hybrid search query
        search_query = {
            "size": size,
            "query": {
                "bool": {
                    "should": [
                        # Keyword search component
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "transcript^2", "summary", "action_items", "decisions"],
                                "type": "best_fields",
                                "boost": self.keyword_weight
                            }
                        },
                        # Semantic search component using kNN
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                    "params": {"query_vector": query_embedding}
                                },
                                "boost": self.semantic_weight
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "highlight": {
                "fields": {
                    "transcript": {"fragment_size": 150, "number_of_fragments": 3},
                    "summary": {"fragment_size": 150, "number_of_fragments": 2}
                }
            }
        }
        
        # Add filters if provided
        if filters:
            search_query["query"]["bool"]["filter"] = self._build_filters(filters)
        
        try:
            response = es_client.search(index=ELASTIC_INDEX, body=search_query)
            results = []
            
            for hit in response['hits']['hits']:
                result = hit['_source']
                result['score'] = hit['_score']
                result['highlights'] = hit.get('highlight', {})
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            return []
    
    def _build_filters(self, filters: Dict) -> List[Dict]:
        """Build Elasticsearch filters from filter dictionary"""
        filter_clauses = []
        
        if 'platform' in filters:
            filter_clauses.append({"term": {"platform": filters['platform']}})
        
        if 'participants' in filters:
            filter_clauses.append({"terms": {"participants": filters['participants']}})
        
        if 'date_range' in filters:
            date_filter = {"range": {"start_time": {}}}
            if 'start' in filters['date_range']:
                date_filter["range"]["start_time"]["gte"] = filters['date_range']['start']
            if 'end' in filters['date_range']:
                date_filter["range"]["start_time"]["lte"] = filters['date_range']['end']
            filter_clauses.append(date_filter)
        
        if 'tags' in filters:
            filter_clauses.append({"terms": {"tags": filters['tags']}})
        
        if 'sentiment' in filters:
            sentiment_filter = {"range": {"sentiment_score": {}}}
            if filters['sentiment'] == 'positive':
                sentiment_filter["range"]["sentiment_score"]["gte"] = 0.5
            elif filters['sentiment'] == 'negative':
                sentiment_filter["range"]["sentiment_score"]["lt"] = 0.5
            filter_clauses.append(sentiment_filter)
        
        return filter_clauses
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for query"""
        try:
            model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
            embeddings = model.get_embeddings([text])
            return embeddings[0].values
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 768

hybrid_search = HybridSearchService()

# ============================================================================
# VERTEX AI AGENTIC LAYER
# ============================================================================

class AgentSpaceService:
    """Vertex AI powered agent for intelligent query processing and RAG"""
    
    def __init__(self):
        self.model = None
        try:
            self.model = GenerativeModel("gemini-1.5-pro")
            logger.info("Vertex AI model initialized")
        except Exception as e:
            logger.error(f"Error initializing Vertex AI model: {e}")
    
    def process_query(self, query: str, user_context: Dict = None) -> Dict:
        """Process natural language query and generate intelligent response"""
        if not self.model:
            return {'error': 'AI model not available'}
        
        try:
            # Step 1: Analyze query intent
            intent = self._analyze_intent(query)
            
            # Step 2: Determine data retrieval strategy
            retrieval_strategy = self._determine_strategy(intent, query)
            
            # Step 3: Retrieve relevant data
            retrieved_data = self._retrieve_data(query, retrieval_strategy)
            
            # Step 4: Generate response using RAG
            response = self._generate_rag_response(query, retrieved_data, intent)
            
            return {
                'query': query,
                'intent': intent,
                'strategy': retrieval_strategy,
                'response': response,
                'sources': retrieved_data.get('sources', []),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {'error': str(e)}
    
    def _analyze_intent(self, query: str) -> Dict:
        """Analyze user query to determine intent"""
        prompt = f"""Analyze the following query and determine the user's intent.
        
Query: {query}

Classify the intent into one of these categories:
- search: Looking for specific meetings or information
- summarize: Wants a summary of meetings or topics
- timeline: Wants chronological information
- analytics: Wants insights, trends, or metrics
- action_items: Wants to see or create action items
- calendar: Related to calendar events or scheduling

Return a JSON with: {{"intent": "category", "entities": ["extracted", "entities"], "time_range": "if applicable"}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            intent_data = json.loads(response.text)
            return intent_data
        except Exception as e:
            logger.error(f"Error analyzing intent: {e}")
            return {'intent': 'search', 'entities': [], 'time_range': None}
    
    def _determine_strategy(self, intent: Dict, query: str) -> str:
        """Determine optimal data retrieval strategy"""
        intent_type = intent.get('intent', 'search')
        
        strategy_map = {
            'search': 'hybrid_search',
            'summarize': 'hybrid_search_aggregate',
            'timeline': 'structured_query',
            'analytics': 'structured_query',
            'action_items': 'structured_query',
            'calendar': 'structured_query'
        }
        
        return strategy_map.get(intent_type, 'hybrid_search')
    
    def _retrieve_data(self, query: str, strategy: str) -> Dict:
        """Retrieve data based on strategy"""
        if strategy == 'hybrid_search':
            results = hybrid_search.search(query, size=15)
            return {
                'type': 'search_results',
                'data': results,
                'sources': [r['meeting_id'] for r in results]
            }
        
        elif strategy == 'hybrid_search_aggregate':
            results = hybrid_search.search(query, size=30)
            return {
                'type': 'aggregated_results',
                'data': results,
                'sources': [r['meeting_id'] for r in results]
            }
        
        elif strategy == 'structured_query':
            sql_results = self._execute_bigquery_query(query)
            return {
                'type': 'structured_data',
                'data': sql_results,
                'sources': ['bigquery']
            }
        
        return {'type': 'unknown', 'data': [], 'sources': []}
    
    def _execute_bigquery_query(self, query: str) -> List[Dict]:
        """Execute structured query on BigQuery"""
        if not bigquery_client:
            return []
        
        # Generate SQL query using AI
        prompt = f"""Convert this natural language query to BigQuery SQL for the meetsync_data dataset.
        
Available tables:
- meetings (meeting_id, title, platform, start_time, end_time, participants, transcript, summary, action_items)
- participants (participant_id, email, name, department, role)
- action_items (action_id, meeting_id, description, assigned_to, due_date, status)
- calendar_events (event_id, title, start_time, end_time, attendees)

Query: {query}

Return only the SQL query, no explanations.
"""
        
        try:
            response = self.model.generate_content(prompt)
            sql_query = response.text.strip().replace('```sql', '').replace('```', '')
            
            query_job = bigquery_client.query(sql_query)
            results = query_job.result()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Error executing BigQuery query: {e}")
            return []
    
    def _generate_rag_response(self, query: str, retrieved_data: Dict, intent: Dict) -> str:
        """Generate response using Retrieval-Augmented Generation"""
        context = self._format_context(retrieved_data)
        
        prompt = f"""You are an AI assistant for MeetSync, a meeting intelligence platform.

User Query: {query}

Context from meetings and data:
{context}

Based on the context above, provide a comprehensive, accurate response to the user's query.
- Be specific and cite relevant meetings when applicable
- Format the response clearly with proper structure
- If asking about trends or patterns, provide insights
- If information is not in the context, say so clearly

Response:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return "I apologize, but I encountered an error generating a response."
    
    def _format_context(self, retrieved_data: Dict) -> str:
        """Format retrieved data as context for LLM"""
        data_type = retrieved_data.get('type', 'unknown')
        data = retrieved_data.get('data', [])
        
        if data_type == 'search_results' or data_type == 'aggregated_results':
            context_parts = []
            for idx, item in enumerate(data[:10], 1):
                context_parts.append(f"""
Meeting {idx}:
- Title: {item.get('title', 'N/A')}
- Date: {item.get('start_time', 'N/A')}
- Platform: {item.get('platform', 'N/A')}
- Participants: {', '.join(item.get('participants', [])[:5])}
- Summary: {item.get('summary', item.get('transcript', '')[:300])}
- Action Items: {', '.join(item.get('action_items', [])[:3])}
""")
            return '\n'.join(context_parts)
        
        elif data_type == 'structured_data':
            return json.dumps(data, indent=2)
        
        return "No relevant data found."
    
    def generate_meeting_summary(self, transcript: str, metadata: Dict) -> Dict:
        """Generate comprehensive meeting summary with AI"""
        prompt = f"""Analyze this meeting transcript and generate a structured summary.

Meeting Details:
- Title: {metadata.get('title', 'N/A')}
- Date: {metadata.get('start_time', 'N/A')}
- Participants: {', '.join(metadata.get('participants', []))}

Transcript:
{transcript[:4000]}

Generate a JSON response with:
{{
  "summary": "Brief 2-3 sentence summary",
  "key_points": ["point1", "point2", ...],
  "decisions": ["decision1", "decision2", ...],
  "action_items": [
    {{"description": "task", "assigned_to": "person", "priority": "high/medium/low"}}
  ],
  "sentiment": "positive/neutral/negative",
  "sentiment_score": 0.0-1.0,
  "topics": ["topic1", "topic2", ...],
  "next_steps": ["step1", "step2", ...]
}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            summary_data = json.loads(response.text)
            return summary_data
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                'summary': 'Error generating summary',
                'key_points': [],
                'decisions': [],
                'action_items': [],
                'sentiment': 'neutral',
                'sentiment_score': 0.5,
                'topics': [],
                'next_steps': []
            }

agent_service = AgentSpaceService()

# ============================================================================
# GOOGLE CALENDAR INTEGRATION
# ============================================================================

class CalendarService:
    """Google Calendar integration for meeting management"""
    
    def __init__(self):
        self.scopes = ['https://www.googleapis.com/auth/calendar']
        self.service = None
    
    def authenticate(self, credentials_dict: Dict):
        """Authenticate with Google Calendar API"""
        try:
            credentials = service_account.Credentials.from_service_account_info(
                credentials_dict, scopes=self.scopes
            )
            self.service = build('calendar', 'v3', credentials=credentials)
            logger.info("Google Calendar authenticated")
        except Exception as e:
            logger.error(f"Calendar authentication error: {e}")
    
    def get_upcoming_meetings(self, user_email: str, days: int = 7) -> List[Dict]:
        """Get upcoming meetings for a user"""
        if not self.service:
            return []
        
        try:
            now = datetime.utcnow().isoformat() + 'Z'
            end_date = (datetime.utcnow() + timedelta(days=days)).isoformat() + 'Z'
            
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=now,
                timeMax=end_date,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            meetings = []
            for event in events:
                if 'conferenceData' in event or 'hangoutLink' in event:
                    meetings.append({
                        'event_id': event['id'],
                        'title': event.get('summary', 'No Title'),
                        'start_time': event['start'].get('dateTime', event['start'].get('date')),
                        'end_time': event['end'].get('dateTime', event['end'].get('date')),
                        'attendees': [att.get('email') for att in event.get('attendees', [])],
                        'meeting_link': event.get('hangoutLink', ''),
                        'description': event.get('description', ''),
                        'location': event.get('location', '')
                    })
            
            return meetings
            
        except Exception as e:
            logger.error(f"Error fetching upcoming meetings: {e}")
            return []
    
    def get_past_meetings(self, user_email: str, days: int = 30) -> List[Dict]:
        """Get past meetings for a user"""
        if not self.service:
            return []
        
        try:
            start_date = (datetime.utcnow() - timedelta(days=days)).isoformat() + 'Z'
            now = datetime.utcnow().isoformat() + 'Z'
            
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=start_date,
                timeMax=now,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            meetings = []
            for event in events:
                if 'conferenceData' in event or 'hangoutLink' in event:
                    meetings.append({
                        'event_id': event['id'],
                        'title': event.get('summary', 'No Title'),
                        'start_time': event['start'].get('dateTime', event['start'].get('date')),
                        'end_time': event['end'].get('dateTime', event['end'].get('date')),
                        'attendees': [att.get('email') for att in event.get('attendees', [])],
                        'meeting_link': event.get('hangoutLink', ''),
                        'status': 'completed'
                    })
            
            return meetings
            
        except Exception as e:
            logger.error(f"Error fetching past meetings: {e}")
            return []
    
    def create_meeting(self, meeting_data: Dict) -> Optional[str]:
        """Create a new calendar event"""
        if not self.service:
            return None
        
        try:
            event = {
                'summary': meeting_data['title'],
                'description': meeting_data.get('description', ''),
                'start': {
                    'dateTime': meeting_data['start_time'],
                    'timeZone': meeting_data.get('timezone', 'UTC')
                },
                'end': {
                    'dateTime': meeting_data['end_time'],
                    'timeZone': meeting_data.get('timezone', 'UTC')
                },
                'attendees': [{'email': email} for email in meeting_data.get('attendees', [])],
                'conferenceData': {
                    'createRequest': {
                        'requestId': str(uuid.uuid4()),
                        'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                    }
                },
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': meeting_data.get('reminder_minutes', 30)},
                        {'method': 'popup', 'minutes': 10}
                    ]
                }
            }
            
            created_event = self.service.events().insert(
                calendarId='primary',
                body=event,
                conferenceDataVersion=1
            ).execute()
            
            logger.info(f"Created calendar event: {created_event['id']}")
            return created_event['id']
            
        except Exception as e:
            logger.error(f"Error creating meeting: {e}")
            return None
    
    def set_reminder(self, event_id: str, reminder_minutes: int) -> bool:
        """Set or update reminder for an event"""
        if not self.service:
            return False
        
        try:
            event = self.service.events().get(calendarId='primary', eventId=event_id).execute()
            
            event['reminders'] = {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': reminder_minutes},
                    {'method': 'popup', 'minutes': 10}
                ]
            }
            
            self.service.events().update(
                calendarId='primary',
                eventId=event_id,
                body=event
            ).execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting reminder: {e}")
            return False

calendar_service = CalendarService()

# ============================================================================
# NOTIFICATION SERVICE
# ============================================================================

class NotificationService:
    """Handle multi-channel notifications (Email, SMS, WhatsApp, Push)"""
    
    def __init__(self):
        self.channels = ['email', 'sms', 'whatsapp', 'push']
    
    def send_notification(self, notification: Dict) -> Dict:
        """Send notification through specified channels"""
        results = {}
        channels = notification.get('channels', ['email'])
        
        for channel in channels:
            if channel == 'email':
                results['email'] = self._send_email(notification)
            elif channel == 'sms':
                results['sms'] = self._send_sms(notification)
            elif channel == 'whatsapp':
                results['whatsapp'] = self._send_whatsapp(notification)
            elif channel == 'push':
                results['push'] = self._send_push(notification)
        
        # Log notification in BigQuery
        self._log_notification(notification, results)
        
        return results
    
    def _send_email(self, notification: Dict) -> bool:
        """Send email notification via SendGrid"""
        if not sg_client:
            return False
        
        try:
            message = Mail(
                from_email=Email('noreply@meetsync.ai'),
                to_emails=To(notification['recipient']),
                subject=notification['title'],
                html_content=Content("text/html", notification['message'])
            )
            
            response = sg_client.send(message)
            logger.info(f"Email sent to {notification['recipient']}: {response.status_code}")
            return response.status_code == 202
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def _send_sms(self, notification: Dict) -> bool:
        """Send SMS notification via Twilio"""
        if not twilio_client:
            return False
        
        try:
            message = twilio_client.messages.create(
                body=notification['message'],
                from_=TWILIO_PHONE_NUMBER,
                to=notification['phone']
            )
            
            logger.info(f"SMS sent to {notification['phone']}: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
            return False
    
    def _send_whatsapp(self, notification: Dict) -> bool:
        """Send WhatsApp message via Twilio"""
        if not twilio_client:
            return False
        
        try:
            message = twilio_client.messages.create(
                body=notification['message'],
                from_=f'whatsapp:{TWILIO_WHATSAPP_NUMBER}',
                to=f'whatsapp:{notification["phone"]}'
            )
            
            logger.info(f"WhatsApp sent to {notification['phone']}: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending WhatsApp: {e}")
            return False
    
    def _send_push(self, notification: Dict) -> bool:
        """Send push notification (placeholder for FCM/APNS integration)"""
        logger.info(f"Push notification: {notification['title']}")
        return True
    
    def _log_notification(self, notification: Dict, results: Dict):
        """Log notification in BigQuery"""
        if not bigquery_client:
            return
        
        table_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.notifications"
        
        row = {
            'notification_id': str(uuid.uuid4()),
            'user_id': notification.get('user_id'),
            'type': notification.get('type'),
            'title': notification.get('title'),
            'message': notification.get('message'),
            'channels': notification.get('channels', []),
            'scheduled_time': notification.get('scheduled_time'),
            'sent_time': datetime.utcnow().isoformat(),
            'status': 'sent' if any(results.values()) else 'failed',
            'metadata': json.dumps(results),
            'created_at': datetime.utcnow().isoformat()
        }
        
        bigquery_client.insert_rows_json(table_id, [row])
    
    def schedule_meeting_reminder(self, meeting: Dict, user: Dict, minutes_before: int):
        """Schedule reminder for upcoming meeting"""
        reminder_time = datetime.fromisoformat(meeting['start_time']) - timedelta(minutes=minutes_before)
        
        notification = {
            'user_id': user['user_id'],
            'recipient': user['email'],
            'phone': user.get('phone'),
            'type': 'meeting_reminder',
            'title': f'Meeting Reminder: {meeting["title"]}',
            'message': f'''
Your meeting "{meeting['title']}" starts in {minutes_before} minutes.

Time: {meeting['start_time']}
Duration: {meeting.get('duration_minutes', 'N/A')} minutes
Link: {meeting.get('meeting_link', 'N/A')}

Participants: {', '.join(meeting.get('participants', [])[:3])}
''',
            'channels': user.get('notification_preferences', ['email']),
            'scheduled_time': reminder_time.isoformat()
        }
        
        # Schedule with APScheduler
        scheduler.add_job(
            func=self.send_notification,
            trigger='date',
            run_date=reminder_time,
            args=[notification],
            id=f"reminder_{meeting['event_id']}_{minutes_before}"
        )
        
        logger.info(f"Scheduled reminder for meeting {meeting['event_id']} at {reminder_time}")
    
    def send_meeting_summary(self, meeting_id: str, summary: Dict, attendees: List[str]):
        """Send automatic meeting summary to attendees"""
        for attendee in attendees:
            notification = {
                'recipient': attendee,
                'type': 'meeting_summary',
                'title': f'Meeting Summary: {summary.get("title", "Meeting")}',
                'message': f'''
Meeting Summary

Date: {summary.get('start_time')}
Duration: {summary.get('duration_minutes')} minutes

Summary:
{summary.get('summary', 'N/A')}

Key Decisions:
{chr(10).join(['- ' + d for d in summary.get('decisions', [])])}

Action Items:
{chr(10).join(['- ' + a['description'] + ' (Assigned to: ' + a.get('assigned_to', 'TBD') + ')' for a in summary.get('action_items', [])])}

Next Steps:
{chr(10).join(['- ' + s for s in summary.get('next_steps', [])])}
''',
                'channels': ['email']
            }
            
            self.send_notification(notification)

notification_service = NotificationService()

# ============================================================================
# COLLABORATION NETWORK ANALYSIS
# ============================================================================

class CollaborationNetworkService:
    """Analyze and visualize collaboration networks"""
    
    def __init__(self):
        self.graph = nx.Graph()
    
    def build_network(self, days: int = 90) -> nx.Graph:
        """Build collaboration network from meeting data"""
        if not bigquery_client:
            return self.graph
        
        query = f"""
        SELECT 
            m.meeting_id,
            m.participants,
            m.start_time,
            m.platform
        FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.meetings` m
        WHERE m.start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
        """
        
        try:
            results = bigquery_client.query(query).result()
            
            # Build network graph
            for row in results:
                participants = row.participants
                
                # Add nodes
                for participant in participants:
                    if not self.graph.has_node(participant):
                        self.graph.add_node(participant, meetings=0, platforms=set())
                    
                    self.graph.nodes[participant]['meetings'] += 1
                    self.graph.nodes[participant]['platforms'].add(row.platform)
                
                # Add edges between all pairs of participants
                for i, p1 in enumerate(participants):
                    for p2 in participants[i+1:]:
                        if self.graph.has_edge(p1, p2):
                            self.graph[p1][p2]['weight'] += 1
                        else:
                            self.graph.add_edge(p1, p2, weight=1)
            
            logger.info(f"Built collaboration network: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            return self.graph
            
        except Exception as e:
            logger.error(f"Error building collaboration network: {e}")
            return self.graph
    
    def get_network_metrics(self) -> Dict:
        """Calculate network metrics"""
        if self.graph.number_of_nodes() == 0:
            return {}
        
        metrics = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_clustering': nx.average_clustering(self.graph),
            'connected_components': nx.number_connected_components(self.graph)
        }
        
        # Calculate centrality measures
        degree_cent = nx.degree_centrality(self.graph)
        betweenness_cent = nx.betweenness_centrality(self.graph)
        
        # Top collaborators
        top_collaborators = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
        
        metrics['top_collaborators'] = [
            {'email': email, 'centrality': score, 'connections': self.graph.degree(email)}
            for email, score in top_collaborators
        ]
        
        # Network bridges (high betweenness)
        bridges = sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:5]
        metrics['bridges'] = [
            {'email': email, 'betweenness': score}
            for email, score in bridges
        ]
        
        return metrics
    
    def visualize_network(self) -> str:
        """Generate interactive network visualization"""
        if self.graph.number_of_nodes() == 0:
            return "{}"
        
        # Calculate layout
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        
        # Prepare edge trace
        edge_trace = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = self.graph[edge[0]][edge[1]]['weight']
            
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=min(weight, 10), color='#888'),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        # Prepare node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            meetings = self.graph.nodes[node].get('meetings', 0)
            platforms = self.graph.nodes[node].get('platforms', set())
            connections = self.graph.degree(node)
            
            node_text.append(f"{node}<br>Meetings: {meetings}<br>Connections: {connections}<br>Platforms: {', '.join(platforms)}")
            node_size.append(10 + meetings * 2)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=edge_trace + [node_trace],
            layout=go.Layout(
                title='Collaboration Network',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
        )
        
        return fig.to_json()
    
    def get_user_collaboration_details(self, user_email: str) -> Dict:
        """Get detailed collaboration info for a specific user"""
        if not self.graph.has_node(user_email):
            return {}
        
        neighbors = list(self.graph.neighbors(user_email))
        
        collaborations = []
        for neighbor in neighbors:
            collaborations.append({
                'email': neighbor,
                'meeting_count': self.graph[user_email][neighbor]['weight'],
                'strength': self.graph[user_email][neighbor]['weight'] / self.graph.nodes[user_email]['meetings']
            })
        
        collaborations.sort(key=lambda x: x['meeting_count'], reverse=True)
        
        return {
            'email': user_email,
            'total_meetings': self.graph.nodes[user_email].get('meetings', 0),
            'total_collaborators': len(neighbors),
            'platforms': list(self.graph.nodes[user_email].get('platforms', set())),
            'top_collaborators': collaborations[:10],
            'centrality': nx.degree_centrality(self.graph).get(user_email, 0)
        }

collaboration_network = CollaborationNetworkService()

# ============================================================================
# ANALYTICS & VISUALIZATION SERVICE
# ============================================================================

class AnalyticsService:
    """Generate analytics and visualizations from meeting data"""
    
    def get_meeting_analytics(self, filters: Dict = None) -> Dict:
        """Get comprehensive meeting analytics"""
        if not bigquery_client:
            return {}
        
        # Base query
        base_conditions = []
        if filters:
            if 'date_start' in filters:
                base_conditions.append(f"start_time >= '{filters['date_start']}'")
            if 'date_end' in filters:
                base_conditions.append(f"start_time <= '{filters['date_end']}'")
            if 'platform' in filters:
                base_conditions.append(f"platform = '{filters['platform']}'")
        
        where_clause = "WHERE " + " AND ".join(base_conditions) if base_conditions else ""
        
        queries = {
            'total_meetings': f"""
                SELECT COUNT(*) as count
                FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.meetings`
                {where_clause}
            """,
            'by_platform': f"""
                SELECT platform, COUNT(*) as count
                FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.meetings`
                {where_clause}
                GROUP BY platform
                ORDER BY count DESC
            """,
            'by_day': f"""
                SELECT DATE(start_time) as date, COUNT(*) as count
                FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.meetings`
                {where_clause}
                GROUP BY date
                ORDER BY date
            """,
            'avg_duration': f"""
                SELECT AVG(duration_minutes) as avg_duration
                FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.meetings`
                {where_clause}
            """,
            'sentiment_distribution': f"""
                SELECT 
                    CASE 
                        WHEN sentiment_score >= 0.7 THEN 'positive'
                        WHEN sentiment_score >= 0.4 THEN 'neutral'
                        ELSE 'negative'
                    END as sentiment,
                    COUNT(*) as count
                FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.meetings` {where_clause}
                GROUP BY sentiment
            """,
            'top_participants': f"""
                SELECT participant, COUNT(*) as meeting_count
                FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.meetings`,
                UNNEST(participants) as participant
                {where_clause}
                GROUP BY participant
                ORDER BY meeting_count DESC
                LIMIT 10
            """
        }
        
        analytics = {}
        
        try:
            for key, query in queries.items():
                result = bigquery_client.query(query).result()
                analytics[key] = [dict(row) for row in result]
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error generating analytics: {e}")
            return {}
    
    def create_timeline_visualization(self, query: str, filters: Dict = None) -> str:
        """Create timeline visualization for feature decisions or meeting themes"""
        if not bigquery_client:
            return "{}"
        
        # Search for relevant meetings
        meetings = hybrid_search.search(query, filters=filters, size=50)
        
        if not meetings:
            return "{}"
        
        # Create timeline data
        timeline_data = []
        for meeting in meetings:
            timeline_data.append({
                'date': meeting.get('start_time', ''),
                'title': meeting.get('title', ''),
                'platform': meeting.get('platform', ''),
                'summary': meeting.get('summary', '')[:100],
                'participants': len(meeting.get('participants', [])),
                'sentiment': meeting.get('sentiment_score', 0.5)
            })
        
        # Sort by date
        timeline_data.sort(key=lambda x: x['date'])
        
        # Create Plotly timeline
        df = pd.DataFrame(timeline_data)
        
        fig = go.Figure()
        
        # Add scatter plot for meetings
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['sentiment'],
            mode='markers+text',
            marker=dict(
                size=df['participants'] * 3,
                color=df['sentiment'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Sentiment")
            ),
            text=df['title'],
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Sentiment: %{y:.2f}<br>Participants: %{marker.size}'
        ))
        
        fig.update_layout(
            title=f'Timeline: {query}',
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            hovermode='closest',
            height=500
        )
        
        return fig.to_json()
    
    def create_dashboard_visualizations(self) -> Dict[str, str]:
        """Create all dashboard visualizations"""
        visualizations = {}
        
        # 1. Meeting frequency over time
        analytics = self.get_meeting_analytics()
        
        if analytics.get('by_day'):
            df_daily = pd.DataFrame(analytics['by_day'])
            fig_daily = px.line(
                df_daily, 
                x='date', 
                y='count',
                title='Meeting Frequency Over Time',
                labels={'count': 'Number of Meetings', 'date': 'Date'}
            )
            visualizations['meeting_frequency'] = fig_daily.to_json()
        
        # 2. Platform distribution
        if analytics.get('by_platform'):
            df_platform = pd.DataFrame(analytics['by_platform'])
            fig_platform = px.pie(
                df_platform,
                values='count',
                names='platform',
                title='Meetings by Platform'
            )
            visualizations['platform_distribution'] = fig_platform.to_json()
        
        # 3. Sentiment distribution
        if analytics.get('sentiment_distribution'):
            df_sentiment = pd.DataFrame(analytics['sentiment_distribution'])
            fig_sentiment = px.bar(
                df_sentiment,
                x='sentiment',
                y='count',
                title='Meeting Sentiment Distribution',
                color='sentiment',
                color_discrete_map={'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
            )
            visualizations['sentiment_distribution'] = fig_sentiment.to_json()
        
        # 4. Top participants
        if analytics.get('top_participants'):
            df_participants = pd.DataFrame(analytics['top_participants'])
            fig_participants = px.bar(
                df_participants,
                x='meeting_count',
                y='participant',
                orientation='h',
                title='Most Active Participants'
            )
            visualizations['top_participants'] = fig_participants.to_json()
        
        return visualizations

analytics_service = AnalyticsService()

# ============================================================================
# SCHEDULED TASKS
# ============================================================================

def scheduled_fivetran_sync():
    """Sync all Fivetran connectors every 6 hours"""
    logger.info("Starting scheduled Fivetran sync")
    connectors = fivetran_manager.list_connectors()
    
    for connector in connectors:
        connector_id = connector.get('id')
        if connector_id:
            fivetran_manager.sync_connector(connector_id)
    
    logger.info(f"Completed sync for {len(connectors)} connectors")

def scheduled_network_update():
    """Update collaboration network daily"""
    logger.info("Updating collaboration network")
    collaboration_network.build_network(days=90)
    logger.info("Collaboration network updated")

def scheduled_cache_warmup():
    """Warm up cache with common queries"""
    if not redis_client:
        return
    
    logger.info("Warming up cache")
    common_queries = [
        "product planning meetings last week",
        "design review meetings",
        "engineering standup",
        "quarterly planning"
    ]
    
    for query in common_queries:
        results = hybrid_search.search(query, size=20)
        cache_key = f"search:{hashlib.md5(query.encode()).hexdigest()}"
        redis_client.setex(cache_key, 3600, json.dumps(results))
    
    logger.info("Cache warmup completed")

# Schedule periodic tasks
scheduler.add_job(scheduled_fivetran_sync, 'interval', hours=6, id='fivetran_sync')
scheduler.add_job(scheduled_network_update, 'interval', hours=24, id='network_update')
scheduler.add_job(scheduled_cache_warmup, 'interval', hours=12, id='cache_warmup')

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('index.html')

# Path: backend/app.py
# Updated login endpoint section only - rest remains the same

# Find the @app.route('/api/auth/login', methods=['POST']) section and replace with:

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User authentication endpoint with TEST MODE support"""
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    logger.info(f"Login attempt for: {email}")
    
    # TEST MODE: Set to True for demo/testing, False for production
    # This can also be controlled by environment variable
    TEST_MODE = os.getenv('TEST_MODE', 'True').lower() == 'true'
    
    if TEST_MODE:
        logger.info("TEST MODE ENABLED: Accepting credentials for testing")
        
        # Generate user data from email
        user_data = {
            'user_id': f'user_{hashlib.md5(email.encode()).hexdigest()[:8]}',
            'email': email,
            'name': email.split('@')[0].replace('.', ' ').title(),
            'department': 'Engineering',
            'role': 'Team Member'
        }
        
        token = generate_token(user_data)
        
        logger.info(f"Generated token for {email} in TEST MODE")
        
        return jsonify({
            'success': True,
            'token': token,
            'user': user_data,
            'mode': 'test'
        })
    
    # PRODUCTION MODE: Verify credentials
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    # Query user from database
    if bigquery_client:
        query = f"""
        SELECT participant_id, email, name, role, department
        FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.participants`
        WHERE email = '{email}'
        LIMIT 1
        """
        
        try:
            result = bigquery_client.query(query).result()
            users = [dict(row) for row in result]
            
            if users:
                user = users[0]
                token = generate_token({
                    'user_id': user['participant_id'],
                    'email': user['email'],
                    'name': user['name']
                })
                
                return jsonify({
                    'success': True,
                    'token': token,
                    'user': user,
                    'mode': 'production'
                })
        except Exception as e:
            logger.error(f"Login error: {e}")
    
    logger.warning(f"Login failed for: {email}")
    return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

@app.route('/api/auth/google', methods=['POST'])
def google_auth():
    """Google OAuth authentication"""
    data = request.json
    id_token = data.get('id_token')
    
    # Verify Google ID token (simplified)
    # In production, use google.oauth2.id_token.verify_oauth2_token
    
    user_data = {
        'user_id': str(uuid.uuid4()),
        'email': data.get('email'),
        'name': data.get('name')
    }
    
    token = generate_token(user_data)
    
    return jsonify({
        'success': True,
        'token': token,
        'user': user_data
    })

@app.route('/api/meetings/ingest', methods=['POST'])
@require_auth
def ingest_meeting():
    """Ingest meeting data from various platforms"""
    data = request.json
    platform = data.get('platform')
    meeting_data = data.get('data')
    
    if not platform or not meeting_data:
        return jsonify({'error': 'Missing platform or data'}), 400
    
    success = ingestion_service.ingest_meeting_data(platform, meeting_data)
    
    if success:
        return jsonify({'success': True, 'message': 'Meeting data ingested successfully'})
    else:
        return jsonify({'success': False, 'error': 'Failed to ingest meeting data'}), 500

@app.route('/api/search', methods=['POST'])
@require_auth
def search_meetings():
    """Hybrid search endpoint"""
    data = request.json
    query = data.get('query', '')
    filters = data.get('filters', {})
    size = data.get('size', 10)
    
    # Check cache first
    cache_key = f"search:{hashlib.md5((query + str(filters)).encode()).hexdigest()}"
    
    if redis_client:
        cached_result = redis_client.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for query: {query}")
            return jsonify({
                'success': True,
                'results': json.loads(cached_result),
                'cached': True
            })
    
    # Perform search
    results = hybrid_search.search(query, filters=filters, size=size)
    
    # Cache results
    if redis_client:
        redis_client.setex(cache_key, 3600, json.dumps(results))
    
    return jsonify({
        'success': True,
        'results': results,
        'cached': False
    })

@app.route('/api/query', methods=['POST'])
@require_auth
def ai_query():
    """Natural language AI query endpoint"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Check cache
    cache_key = f"ai_query:{hashlib.md5(query.encode()).hexdigest()}"
    
    if redis_client:
        cached_result = redis_client.get(cache_key)
        if cached_result:
            return jsonify(json.loads(cached_result))
    
    # Process query with AI agent
    response = agent_service.process_query(query, user_context=request.user)
    
    # Cache for 1 hour
    if redis_client:
        redis_client.setex(cache_key, 3600, json.dumps(response))
    
    return jsonify(response)

@app.route('/api/meetings/<meeting_id>/summary', methods=['GET'])
@require_auth
def get_meeting_summary(meeting_id):
    """Get AI-generated summary for a specific meeting"""
    if not bigquery_client:
        return jsonify({'error': 'Database not available'}), 503
    
    query = f"""
    SELECT *
    FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.meetings`
    WHERE meeting_id = '{meeting_id}'
    LIMIT 1
    """
    
    try:
        result = bigquery_client.query(query).result()
        meetings = [dict(row) for row in result]
        
        if not meetings:
            return jsonify({'error': 'Meeting not found'}), 404
        
        meeting = meetings[0]
        
        # Generate summary if not already present
        if not meeting.get('summary'):
            summary = agent_service.generate_meeting_summary(
                meeting.get('transcript', ''),
                meeting
            )
            
            # Update in database
            update_query = f"""
            UPDATE `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.meetings`
            SET 
                summary = '{summary["summary"]}',
                action_items = {json.dumps(summary["action_items"])},
                decisions = {json.dumps(summary["decisions"])},
                sentiment_score = {summary["sentiment_score"]}
            WHERE meeting_id = '{meeting_id}'
            """
            bigquery_client.query(update_query).result()
            
            meeting['summary'] = summary['summary']
            meeting['action_items'] = summary['action_items']
            meeting['decisions'] = summary['decisions']
            meeting['sentiment_score'] = summary['sentiment_score']
        
        return jsonify({
            'success': True,
            'meeting': meeting
        })
        
    except Exception as e:
        logger.error(f"Error fetching meeting summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/calendar/upcoming', methods=['GET'])
@require_auth
def get_upcoming_meetings():
    """Get upcoming meetings from Google Calendar"""
    days = request.args.get('days', 7, type=int)
    user_email = request.user.get('email')
    
    meetings = calendar_service.get_upcoming_meetings(user_email, days)
    
    return jsonify({
        'success': True,
        'meetings': meetings,
        'count': len(meetings)
    })

@app.route('/api/calendar/past', methods=['GET'])
@require_auth
def get_past_meetings():
    """Get past meetings from Google Calendar"""
    days = request.args.get('days', 30, type=int)
    user_email = request.user.get('email')
    
    meetings = calendar_service.get_past_meetings(user_email, days)
    
    return jsonify({
        'success': True,
        'meetings': meetings,
        'count': len(meetings)
    })

@app.route('/api/calendar/create', methods=['POST'])
@require_auth
def create_meeting():
    """Create a new calendar meeting"""
    data = request.json
    
    event_id = calendar_service.create_meeting(data)
    
    if event_id:
        # Schedule reminders if specified
        if data.get('reminders'):
            for reminder_minutes in data['reminders']:
                notification_service.schedule_meeting_reminder(
                    {'event_id': event_id, **data},
                    request.user,
                    reminder_minutes
                )
        
        return jsonify({
            'success': True,
            'event_id': event_id,
            'message': 'Meeting created successfully'
        })
    else:
        return jsonify({'success': False, 'error': 'Failed to create meeting'}), 500

@app.route('/api/calendar/reminder', methods=['POST'])
@require_auth
def set_meeting_reminder():
    """Set reminder for a calendar event"""
    data = request.json
    event_id = data.get('event_id')
    reminder_minutes = data.get('reminder_minutes', 30)
    
    success = calendar_service.set_reminder(event_id, reminder_minutes)
    
    if success:
        return jsonify({'success': True, 'message': 'Reminder set successfully'})
    else:
        return jsonify({'success': False, 'error': 'Failed to set reminder'}), 500

@app.route('/api/notifications/send', methods=['POST'])
@require_auth
def send_notification():
    """Send notification through specified channels"""
    data = request.json
    
    results = notification_service.send_notification(data)
    
    return jsonify({
        'success': any(results.values()),
        'results': results
    })

@app.route('/api/notifications/preferences', methods=['POST'])
@require_auth
def update_notification_preferences():
    """Update user notification preferences"""
    data = request.json
    user_id = request.user.get('user_id')
    
    preferences = {
        'email': data.get('email', True),
        'sms': data.get('sms', False),
        'whatsapp': data.get('whatsapp', False),
        'push': data.get('push', True),
        'reminder_times': data.get('reminder_times', [30, 10])
    }
    
    # Update in database
    if bigquery_client:
        update_query = f"""
        UPDATE `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.participants`
        SET notification_preferences = '{json.dumps(preferences)}'
        WHERE participant_id = '{user_id}'
        """
        bigquery_client.query(update_query).result()
    
    return jsonify({
        'success': True,
        'preferences': preferences
    })

@app.route('/api/analytics', methods=['GET'])
@require_auth
def get_analytics():
    """Get meeting analytics"""
    filters = {
        'date_start': request.args.get('date_start'),
        'date_end': request.args.get('date_end'),
        'platform': request.args.get('platform')
    }
    
    # Remove None values
    filters = {k: v for k, v in filters.items() if v is not None}
    
    analytics = analytics_service.get_meeting_analytics(filters)
    
    return jsonify({
        'success': True,
        'analytics': analytics
    })

@app.route('/api/analytics/visualizations', methods=['GET'])
@require_auth
def get_visualizations():
    """Get dashboard visualizations"""
    visualizations = analytics_service.create_dashboard_visualizations()
    
    return jsonify({
        'success': True,
        'visualizations': visualizations
    })

@app.route('/api/analytics/timeline', methods=['POST'])
@require_auth
def get_timeline():
    """Get timeline visualization for specific query"""
    data = request.json
    query = data.get('query', '')
    filters = data.get('filters', {})
    
    timeline = analytics_service.create_timeline_visualization(query, filters)
    
    return jsonify({
        'success': True,
        'timeline': timeline
    })

@app.route('/api/network', methods=['GET'])
@require_auth
def get_collaboration_network():
    """Get collaboration network data"""
    days = request.args.get('days', 90, type=int)
    
    # Build or update network
    collaboration_network.build_network(days)
    
    # Get metrics
    metrics = collaboration_network.get_network_metrics()
    
    # Get visualization
    visualization = collaboration_network.visualize_network()
    
    return jsonify({
        'success': True,
        'metrics': metrics,
        'visualization': visualization
    })

@app.route('/api/network/user/<user_email>', methods=['GET'])
@require_auth
def get_user_network(user_email):
    """Get collaboration details for specific user"""
    details = collaboration_network.get_user_collaboration_details(user_email)
    
    if details:
        return jsonify({
            'success': True,
            'details': details
        })
    else:
        return jsonify({'error': 'User not found in network'}), 404

@app.route('/api/action-items', methods=['GET'])
@require_auth
def get_action_items():
    """Get action items for user"""
    user_email = request.user.get('email')
    status = request.args.get('status', 'pending')
    
    if not bigquery_client:
        return jsonify({'error': 'Database not available'}), 503
    
    query = f"""
    SELECT *
    FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.action_items`
    WHERE assigned_to = '{user_email}' AND status = '{status}'
    ORDER BY due_date ASC
    """
    
    try:
        result = bigquery_client.query(query).result()
        action_items = [dict(row) for row in result]
        
        return jsonify({
            'success': True,
            'action_items': action_items,
            'count': len(action_items)
        })
        
    except Exception as e:
        logger.error(f"Error fetching action items: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/action-items/<action_id>/complete', methods=['POST'])
@require_auth
def complete_action_item(action_id):
    """Mark action item as complete"""
    if not bigquery_client:
        return jsonify({'error': 'Database not available'}), 503
    
    update_query = f"""
    UPDATE `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.action_items`
    SET 
        status = 'completed',
        completed_at = CURRENT_TIMESTAMP()
    WHERE action_id = '{action_id}'
    """
    
    try:
        bigquery_client.query(update_query).result()
        
        return jsonify({
            'success': True,
            'message': 'Action item marked as complete'
        })
        
    except Exception as e:
        logger.error(f"Error completing action item: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/fivetran/connectors', methods=['GET'])
@require_auth
def list_connectors():
    """List all Fivetran connectors"""
    connectors = fivetran_manager.list_connectors()
    
    return jsonify({
        'success': True,
        'connectors': connectors,
        'count': len(connectors)
    })

@app.route('/api/fivetran/connectors/<connector_id>/sync', methods=['POST'])
@require_auth
def sync_connector(connector_id):
    """Trigger manual sync for a connector"""
    success = fivetran_manager.sync_connector(connector_id)
    
    if success:
        return jsonify({'success': True, 'message': 'Sync triggered successfully'})
    else:
        return jsonify({'success': False, 'error': 'Failed to trigger sync'}), 500

@app.route('/api/fivetran/connectors/<connector_id>/status', methods=['GET'])
@require_auth
def get_connector_status(connector_id):
    """Get status of a specific connector"""
    status = fivetran_manager.get_connector_status(connector_id)
    
    return jsonify({
        'success': True,
        'status': status
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    services_status = {
        'bigquery': bigquery_client is not None,
        'elasticsearch': es_client is not None and es_client.ping(),
        'redis': redis_client is not None,
        'vertex_ai': agent_service.model is not None,
        'sendgrid': sg_client is not None,
        'twilio': twilio_client is not None
    }
    
    all_healthy = all(services_status.values())
    
    return jsonify({
        'status': 'healthy' if all_healthy else 'degraded',
        'services': services_status,
        'timestamp': datetime.utcnow().isoformat()
    }), 200 if all_healthy else 503

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(error):
    logger.error(f"Unhandled exception: {error}")
    return jsonify({'error': 'An unexpected error occurred'}), 500

# ============================================================================
# APPLICATION STARTUP
# ============================================================================

if __name__ == '__main__':
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("MeetSync AI - Starting Application")
    logger.info("=" * 80)
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 8080)),
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    )
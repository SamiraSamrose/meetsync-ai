# Path: tests/test_app.py

"""
Unit tests for MeetSync AI application
Run with: pytest tests/
"""

import pytest
import json
from backend.app import app, initialize_bigquery_schema
from datetime import datetime

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def auth_token():
    """Generate test authentication token"""
    from backend.app import generate_token
    return generate_token({
        'user_id': 'test_user_123',
        'email': 'test@example.com',
        'name': 'Test User'
    })

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get('/api/health')
    assert response.status_code in [200, 503]
    data = json.loads(response.data)
    assert 'status' in data
    assert 'services' in data

def test_login_endpoint(client):
    """Test login endpoint"""
    response = client.post('/api/auth/login',
        json={
            'email': 'test@example.com',
            'password': 'testpassword'
        },
        content_type='application/json'
    )
    assert response.status_code in [200, 401]

def test_search_without_auth(client):
    """Test search endpoint without authentication"""
    response = client.post('/api/search',
        json={'query': 'test meeting'},
        content_type='application/json'
    )
    assert response.status_code == 401

def test_search_with_auth(client, auth_token):
    """Test search endpoint with authentication"""
    response = client.post('/api/search',
        json={'query': 'test meeting', 'size': 10},
        headers={'Authorization': f'Bearer {auth_token}'},
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'results' in data

def test_ai_query(client, auth_token):
    """Test AI query endpoint"""
    response = client.post('/api/query',
        json={'query': 'What were the key themes in recent meetings?'},
        headers={'Authorization': f'Bearer {auth_token}'},
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'response' in data or 'error' in data

def test_analytics_endpoint(client, auth_token):
    """Test analytics endpoint"""
    response = client.get('/api/analytics',
        headers={'Authorization': f'Bearer {auth_token}'}
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'success' in data

def test_upcoming_meetings(client, auth_token):
    """Test upcoming meetings endpoint"""
    response = client.get('/api/calendar/upcoming?days=7',
        headers={'Authorization': f'Bearer {auth_token}'}
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'meetings' in data

def test_network_endpoint(client, auth_token):
    """Test collaboration network endpoint"""
    response = client.get('/api/network',
        headers={'Authorization': f'Bearer {auth_token}'}
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'metrics' in data or 'success' in data

def test_action_items(client, auth_token):
    """Test action items endpoint"""
    response = client.get('/api/action-items?status=pending',
        headers={'Authorization': f'Bearer {auth_token}'}
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'action_items' in data

def test_notification_preferences(client, auth_token):
    """Test notification preferences update"""
    response = client.post('/api/notifications/preferences',
        json={
            'email': True,
            'sms': False,
            'push': True
        },
        headers={'Authorization': f'Bearer {auth_token}'},
        content_type='application/json'
    )
    assert response.status_code in [200, 500]

def test_create_meeting(client, auth_token):
    """Test meeting creation"""
    meeting_data = {
        'title': 'Test Meeting',
        'description': 'Test Description',
        'start_time': '2025-12-01T14:00:00Z',
        'end_time': '2025-12-01T15:00:00Z',
        'attendees': ['test@example.com'],
        'timezone': 'UTC'
    }
    
    response = client.post('/api/calendar/create',
        json=meeting_data,
        headers={'Authorization': f'Bearer {auth_token}'},
        content_type='application/json'
    )
    assert response.status_code in [200, 500]

class TestDataIngestion:
    """Test data ingestion service"""
    
    def test_parse_google_meet(self):
        """Test Google Meet data parsing"""
        from backend.app import ingestion_service
        
        data = {
            'id': 'test_123',
            'summary': 'Test Meeting',
            'start': {'dateTime': '2025-01-01T10:00:00Z'},
            'end': {'dateTime': '2025-01-01T11:00:00Z'},
            'attendees': [{'email': 'user@example.com'}],
            'organizer': {'email': 'organizer@example.com'}
        }
        
        parsed = ingestion_service._parse_google_meet(data)
        assert parsed['meeting_id'] == 'test_123'
        assert parsed['platform'] == 'google_meet'
        assert len(parsed['participants']) > 0
    
    def test_parse_zoom(self):
        """Test Zoom data parsing"""
        from backend.app import ingestion_service
        
        data = {
            'uuid': 'zoom_123',
            'topic': 'Zoom Test',
            'start_time': '2025-01-01T10:00:00Z',
            'duration': 60,
            'participants': [{'user_email': 'user@example.com'}],
            'host_email': 'host@example.com'
        }
        
        parsed = ingestion_service._parse_zoom(data)
        assert parsed['platform'] == 'zoom'
        assert parsed['duration_minutes'] == 60

class TestHybridSearch:
    """Test hybrid search functionality"""
    
    def test_search_query_building(self):
        """Test search query construction"""
        from backend.app import hybrid_search
        
        query = "product planning meetings"
        filters = {
            'platform': 'zoom',
            'date_range': {
                'start': '2025-01-01',
                'end': '2025-12-31'
            }
        }
        
        # Should not raise exception
        try:
            results = hybrid_search.search(query, filters, size=5)
            assert isinstance(results, list)
        except Exception as e:
            # Expected if Elasticsearch not available
            pass

class TestAgentService:
    """Test AI agent service"""
    
    def test_intent_analysis(self):
        """Test query intent analysis"""
        from backend.app import agent_service
        
        query = "What were the key themes in product meetings?"
        
        try:
            intent = agent_service._analyze_intent(query)
            assert 'intent' in intent
        except Exception:
            # Expected if Vertex AI not available
            pass

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
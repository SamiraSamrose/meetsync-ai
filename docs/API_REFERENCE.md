# Path: docs/API_REFERENCE.md

# MeetSync AI - API Reference

Complete API documentation for MeetSync AI platform.

## Base URL
```
Production: https://your-domain.com
Development: http://localhost:8080
```

## Authentication

All API endpoints (except `/api/health` and `/api/auth/*`) require JWT authentication.

### Headers
```
Authorization: Bearer YOUR_JWT_TOKEN
Content-Type: application/json
```

### Obtain Token

**POST** `/api/auth/login`

Request:
```json
{
  "email": "user@example.com",
  "password": "your_password"
}
```

Response:
```json
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "user_id": "user_123",
    "email": "user@example.com",
    "name": "John Doe"
  }
}
```

---

## Meetings API

### Ingest Meeting Data

**POST** `/api/meetings/ingest`

Ingest meeting data from various platforms.

Request:
```json
{
  "platform": "google_meet",
  "data": {
    "id": "meeting_123",
    "summary": "Product Planning",
    "start": {"dateTime": "2024-01-01T10:00:00Z"},
    "end": {"dateTime": "2024-01-01T11:00:00Z"},
    "attendees": [{"email": "user@example.com"}],
    "transcript": "Meeting transcript..."
  }
}
```

Response:
```json
{
  "success": true,
  "message": "Meeting data ingested successfully"
}
```

### Get Meeting Summary

**GET** `/api/meetings/{meeting_id}/summary`

Get AI-generated summary for a specific meeting.

Response:
```json
{
  "success": true,
  "meeting": {
    "meeting_id": "meeting_123",
    "title": "Product Planning",
    "summary": "The team discussed Q1 priorities...",
    "action_items": [
      {"description": "Update roadmap", "assigned_to": "alice@company.com"}
    ],
    "decisions": ["Approved feature X"],
    "sentiment_score": 0.75
  }
}
```

---

## Search API

### Hybrid Search

**POST** `/api/search`

Perform hybrid search combining keyword and semantic matching.

Request:
```json
{
  "query": "product planning meetings",
  "filters": {
    "platform": "zoom",
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-12-31"
    },
    "participants": ["alice@company.com"]
  },
  "size": 20
}
```

Response:
```json
{
  "success": true,
  "results": [
    {
      "meeting_id": "meeting_123",
      "title": "Product Planning",
      "platform": "zoom",
      "start_time": "2024-01-01T10:00:00Z",
      "participants": ["alice@company.com", "bob@company.com"],
      "summary": "Discussion summary...",
      "score": 0.95,
      "highlights": {
        "transcript": ["...key discussion points..."]
      }
    }
  ],
  "cached": false
}
```

---

## AI Query API

### Natural Language Query

**POST** `/api/query`

Process natural language queries using AI agent.

Request:
```json
{
  "query": "What were the key themes in product planning meetings from July to September?"
}
```

Response:
```json
{
  "query": "What were the key themes...",
  "intent": {
    "intent": "search",
    "entities": ["product planning", "July", "September"]
  },
  "strategy": "hybrid_search_aggregate",
  "response": "Based on the meetings from July to September, the key themes were...",
  "sources": ["meeting_123", "meeting_456"],
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

## Calendar API

### Get Upcoming Meetings

**GET** `/api/calendar/upcoming?days=7`

Get upcoming meetings from Google Calendar.

Response:
```json
{
  "success": true,
  "meetings": [
    {
      "event_id": "event_123",
      "title": "Weekly Sync",
      "start_time": "2024-01-15T14:00:00Z",
      "end_time": "2024-01-15T15:00:00Z",
      "attendees": ["user@example.com"],
      "meeting_link": "https://meet.google.com/abc-defg-hij"
    }
  ],
  "count": 5
}
```

### Get Past Meetings

**GET** `/api/calendar/past?days=30`

Get past meetings from the last N days.

### Create Meeting

**POST** `/api/calendar/create`

Create a new calendar meeting with automatic reminders.

Request:
```json
{
  "title": "Q1 Planning",
  "description": "Quarterly planning session",
  "start_time": "2024-12-01T14:00:00Z",
  "end_time": "2024-12-01T15:30:00Z",
  "attendees": ["user1@example.com", "user2@example.com"],
  "timezone": "America/Los_Angeles",
  "reminders": [30, 60, 1440]
}
```

Response:
```json
{
  "success": true,
  "event_id": "event_123",
  "message": "Meeting created successfully"
}
```

### Set Reminder

**POST** `/api/calendar/reminder`

Set or update reminder for a meeting.

Request:
```json
{
  "event_id": "event_123",
  "reminder_minutes": 30
}
```

---

## Analytics API

### Get Analytics

**GET** `/api/analytics?date_start=2024-01-01&date_end=2024-12-31&platform=zoom`

Get comprehensive meeting analytics.

Response:
```json
{
  "success": true,
  "analytics": {
    "total_meetings": [{"count": 150}],
    "by_platform": [
      {"platform": "zoom", "count": 80},
      {"platform": "google_meet", "count": 70}
    ],
    "avg_duration": [{"avg_duration": 45.5}],
    "sentiment_distribution": [
      {"sentiment": "positive", "count": 90},
      {"sentiment": "neutral", "count": 40},
      {"sentiment": "negative", "count": 20}
    ]
  }
}
```

### Get Visualizations

**GET** `/api/analytics/visualizations`

Get pre-generated Plotly visualizations.

Response:
```json
{
  "success": true,
  "visualizations": {
    "meeting_frequency": "{\"data\": [...], \"layout\": {...}}",
    "platform_distribution": "{\"data\": [...], \"layout\": {...}}",
    "sentiment_distribution": "{\"data\": [...], \"layout\": {...}}"
  }
}
```

### Generate Timeline

**POST** `/api/analytics/timeline`

Generate timeline visualization for specific query.

Request:
```json
{
  "query": "feature decisions design org",
  "filters": {
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-03-31"
    }
  }
}
```

---

## Network API

### Get Collaboration Network

**GET** `/api/network?days=90`

Get collaboration network data and metrics.

Response:
```json
{
  "success": true,
  "metrics": {
    "total_nodes": 25,
    "total_edges": 150,
    "density": 0.48,
    "avg_clustering": 0.62,
    "top_collaborators": [
      {
        "email": "alice@company.com",
        "centrality": 0.85,
        "connections": 18
      }
    ]
  },
  "visualization": "{\"data\": [...], \"layout\": {...}}"
}
```

### Get User Network Details

**GET** `/api/network/user/{email}`

Get detailed collaboration information for specific user.

Response:
```json
{
  "success": true,
  "details": {
    "email": "alice@company.com",
    "total_meetings": 45,
    "total_collaborators": 18,
    "platforms": ["zoom", "google_meet"],
    "top_collaborators": [
      {
        "email": "bob@company.com",
        "meeting_count": 25,
        "strength": 0.56
      }
    ]
  }
}
```

---

## Action Items API

### Get Action Items

**GET** `/api/action-items?status=pending`

Get action items for the authenticated user.

Response:
```json
{
  "success": true,
  "action_items": [
    {
      "action_id": "action_123",
      "meeting_id": "meeting_456",
      "description": "Update product roadmap",
      "assigned_to": "alice@company.com",
      "due_date": "2024-01-15",
      "status": "pending",
      "priority": "high"
    }
  ],
  "count": 5
}
```

### Complete Action Item

**POST** `/api/action-items/{action_id}/complete`

Mark an action item as completed.

Response:
```json
{
  "success": true,
  "message": "Action item marked as complete"
}
```

---

## Notifications API

### Send Notification

**POST** `/api/notifications/send`

Send notification through specified channels.

Request:
```json
{
  "recipient": "user@example.com",
  "phone": "+1234567890",
  "title": "Meeting Reminder",
  "message": "Your meeting starts in 30 minutes",
  "channels": ["email", "sms"],
  "type": "meeting_reminder"
}
```

### Update Notification Preferences

**POST** `/api/notifications/preferences`

Update user notification preferences.

Request:
```json
{
  "email": true,
  "sms": false,
  "whatsapp": true,
  "push": true,
  "reminder_times": [30, 10]
}
```

---

## Fivetran API

### List Connectors

**GET** `/api/fivetran/connectors`

List all Fivetran connectors.

### Sync Connector

**POST** `/api/fivetran/connectors/{connector_id}/sync`

Trigger manual sync for a connector.

### Get Connector Status

**GET** `/api/fivetran/connectors/{connector_id}/status`

Get status of a specific connector.

---

## System API

### Health Check

**GET** `/api/health`

Check system health and service availability.

Response:
```json
{
  "status": "healthy",
  "services": {
    "bigquery": true,
    "elasticsearch": true,
    "redis": true,
    "vertex_ai": true,
    "sendgrid": true,
    "twilio": true
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

## Error Responses

All endpoints may return error responses in the following format:
```json
{
  "error": "Error description",
  "success": false
}
```

Common HTTP Status Codes:
- `200`: Success
- `401`: Unauthorized (missing or invalid token)
- `404`: Resource not found
- `500`: Internal server error
- `503`: Service unavailable
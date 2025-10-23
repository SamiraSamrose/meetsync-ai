# Path: scripts/sample_data_generator.py

"""
Generate sample meeting data for testing
Run with: python scripts/sample_data_generator.py
"""

import json
import random
from datetime import datetime, timedelta
import requests

# Sample data
PLATFORMS = ['google_meet', 'zoom', 'microsoft_teams', 'slack']
PARTICIPANTS = [
    'alice@company.com', 'bob@company.com', 'charlie@company.com',
    'diana@company.com', 'eve@company.com', 'frank@company.com',
    'grace@company.com', 'henry@company.com', 'iris@company.com'
]
TOPICS = [
    'Product Planning', 'Sprint Review', 'Design Review',
    'Engineering Standup', 'Quarterly Planning', 'Customer Feedback',
    'Architecture Discussion', 'Marketing Strategy', 'Sales Pipeline'
]
SENTIMENTS = ['positive', 'neutral', 'negative']

def generate_meeting():
    """Generate a single meeting"""
    start_time = datetime.now() - timedelta(days=random.randint(1, 90))
    duration = random.choice([30, 45, 60, 90])
    
    num_participants = random.randint(3, 8)
    meeting_participants = random.sample(PARTICIPANTS, num_participants)
    
    sentiment = random.choice(SENTIMENTS)
    sentiment_score = {
        'positive': random.uniform(0.6, 1.0),
        'neutral': random.uniform(0.4, 0.6),
        'negative': random.uniform(0.0, 0.4)
    }[sentiment]
    
    meeting = {
        'id': f"meeting_{random.randint(10000, 99999)}",
        'platform': random.choice(PLATFORMS),
        'title': random.choice(TOPICS),
        'start_time': start_time.isoformat(),
        'end_time': (start_time + timedelta(minutes=duration)).isoformat(),
        'duration_minutes': duration,
        'participants': meeting_participants,
        'organizer': meeting_participants[0],
        'transcript': f"This is a transcript of the {random.choice(TOPICS)} meeting discussing various important topics.",
        'summary': f"The team discussed {random.choice(['progress', 'challenges', 'next steps', 'priorities'])} and made several key decisions.",
        'sentiment_score': sentiment_score,
        'action_items': [
            f"Action item {i+1}: Follow up on {random.choice(['feature', 'bug', 'design', 'documentation'])}"
            for i in range(random.randint(1, 4))
        ],
        'decisions': [
            f"Decision {i+1}: Decided to {random.choice(['proceed with', 'postpone', 'approve', 'revise'])} the proposal"
            for i in range(random.randint(1, 3))
        ],
        'tags': [random.choice(['urgent', 'planning', 'review', 'sync', 'brainstorm'])]
    }
    
    return meeting

def generate_sample_data(num_meetings=50):
    """Generate sample dataset"""
    meetings = [generate_meeting() for _ in range(num_meetings)]
    
    # Save to file
    with open('data/sample_meetings.json', 'w') as f:
        json.dump(meetings, f, indent=2)
    
    print(f"Generated {num_meetings} sample meetings")
    print(f"Saved to: data/sample_meetings.json")
    
    return meetings

def upload_to_api(api_url, auth_token, meetings):
    """Upload sample data to API"""
    headers = {'Authorization': f'Bearer {auth_token}'}
    
    for meeting in meetings:
        response = requests.post(
            f"{api_url}/api/meetings/ingest",
            json={'platform': meeting['platform'], 'data': meeting},
            headers=headers
        )
        
        if response.status_code == 200:
            print(f"✓ Uploaded: {meeting['title']}")
        else:
            print(f"✗ Failed: {meeting['title']} - {response.status_code}")

if __name__ == '__main__':
    import sys
    
    num_meetings = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    meetings = generate_sample_data(num_meetings)
    
    # Uncomment to upload to API
    # api_url = "http://localhost:8080"
    # auth_token = "your_token_here"
    # upload_to_api(api_url, auth_token, meetings)
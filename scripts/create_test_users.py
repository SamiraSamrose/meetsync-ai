# Path: scripts/create_test_users.py

"""
Create test users in BigQuery
Run with: python scripts/create_test_users.py
"""

from google.cloud import bigquery
import os

# Configuration
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'your-gcp-project')
BIGQUERY_DATASET = os.getenv('BIGQUERY_DATASET', 'meetsync_data')

# Initialize BigQuery client
client = bigquery.Client(project=GCP_PROJECT_ID)

# Test users
test_users = [
    {
        'participant_id': 'user_001',
        'email': 'alice@company.com',
        'name': 'Alice Johnson',
        'department': 'Engineering',
        'role': 'Senior Engineer',
        'timezone': 'America/Los_Angeles',
        'notification_preferences': '{"email": true, "sms": false, "push": true}'
    },
    {
        'participant_id': 'user_002',
        'email': 'bob@company.com',
        'name': 'Bob Smith',
        'department': 'Product',
        'role': 'Product Manager',
        'timezone': 'America/New_York',
        'notification_preferences': '{"email": true, "sms": true, "push": true}'
    },
    {
        'participant_id': 'user_003',
        'email': 'charlie@company.com',
        'name': 'Charlie Brown',
        'department': 'Design',
        'role': 'Lead Designer',
        'timezone': 'Europe/London',
        'notification_preferences': '{"email": true, "sms": false, "push": false}'
    },
    {
        'participant_id': 'user_004',
        'email': 'diana@company.com',
        'name': 'Diana Prince',
        'department': 'Marketing',
        'role': 'Marketing Manager',
        'timezone': 'America/Chicago',
        'notification_preferences': '{"email": true, "sms": false, "push": true}'
    },
    {
        'participant_id': 'user_005',
        'email': 'admin@company.com',
        'name': 'Admin User',
        'department': 'IT',
        'role': 'Administrator',
        'timezone': 'UTC',
        'notification_preferences': '{"email": true, "sms": true, "push": true}'
    }
]

# Insert users
table_id = f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.participants"

try:
    errors = client.insert_rows_json(table_id, test_users)
    
    if errors:
        print("Errors occurred while inserting test users:")
        print(errors)
    else:
        print(f"Successfully created {len(test_users)} test users!")
        print("\nTest Credentials:")
        print("=" * 60)
        for user in test_users:
            print(f"Email: {user['email']}")
            print(f"Name: {user['name']}")
            print(f"Department: {user['department']}")
            print("-" * 60)
        
        print("\nYou can login with any of these emails and any password")
        print("(Password validation is simplified for demo purposes)")
        
except Exception as e:
    print(f"Error creating test users: {e}")
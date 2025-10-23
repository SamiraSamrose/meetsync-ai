# Path: scripts/setup.sh

#!/bin/bash

# MeetSync AI - Automated Setup Script

echo "========================================="
echo "MeetSync AI - Setup Script"
echo "========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python 3.9+ required. Found: $python_version"
    exit 1
fi

echo "Python version OK: $python_version"

# Create directories
echo "Creating directory structure..."
mkdir -p backend frontend/templates frontend/static config logs data

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r config/requirements.txt

# Create .env template if not exists
if [ ! -f .env ]; then
    echo "Creating .env template..."
    cat > .env << EOF
# Flask Configuration
SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
FLASK_DEBUG=False

# Google Cloud Platform
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=us-central1
GCP_CREDENTIALS_PATH=config/gcp-credentials.json
BIGQUERY_DATASET=meetsync_data
GCS_BUCKET=meetsync-storage

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

# SendGrid
SENDGRID_API_KEY=your-sendgrid-api-key

# Twilio
TWILIO_ACCOUNT_SID=your-twilio-account-sid
TWILIO_AUTH_TOKEN=your-twilio-auth-token
TWILIO_PHONE_NUMBER=+1234567890
TWILIO_WHATSAPP_NUMBER=+1234567890

# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# JWT
JWT_SECRET=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
EOF

    echo ".env file created. Please update with your actual credentials."
fi

echo "========================================="
echo "Setup completed successfully!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Update .env file with your credentials"
echo "2. Set up Google Cloud Platform resources"
echo "3. Configure Fivetran connectors"
echo "4. Run: python backend/app.py"
echo ""
echo "For detailed instructions, see README.md"
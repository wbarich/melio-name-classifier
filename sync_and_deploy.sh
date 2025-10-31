#!/bin/bash

# Sync and Deploy Script for Melio Name Classifier
# This script syncs your project to an EC2 instance and deploys it using docker-compose
#
# Default behavior: Uses the 'melio-ec2' SSH profile from ~/.ssh/config
# Simply run: ./sync_and_deploy.sh (no arguments needed!)

set -e  # Exit on error

# Configuration - Update these values for your EC2 instance
EC2_HOST="${EC2_HOST:-}"
EC2_USER="${EC2_USER:-ubuntu}"
EC2_KEY="${EC2_KEY:-}"
SSH_PROFILE="${SSH_PROFILE:-melio-ec2}"  # Default SSH profile
REMOTE_DIR="${REMOTE_DIR:-~/melio}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Syncs your project to EC2 and deploys it using docker-compose"
    echo ""
    echo "Options:"
    echo "  -p, --profile PROFILE  SSH profile/alias from ~/.ssh/config (recommended)"
    echo "  -h, --host HOST        EC2 instance IP or hostname (required if no profile)"
    echo "  -u, --user USER        SSH username (default: ubuntu)"
    echo "  -k, --key KEY          Path to SSH private key file (required if no profile)"
    echo "  -d, --dir DIR          Remote directory path (default: ~/melio)"
    echo "  --no-deploy            Only sync files, don't deploy"
    echo "  --help                 Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  SSH_PROFILE           SSH profile/alias from ~/.ssh/config"
    echo "  EC2_HOST              EC2 instance IP or hostname"
    echo "  EC2_USER              SSH username"
    echo "  EC2_KEY               Path to SSH private key file"
    echo "  REMOTE_DIR            Remote directory path"
    echo ""
    echo "Examples:"
    echo "  $0                                         # Use default SSH profile (melio-ec2)"
    echo "  $0 -p melio-ec2                            # Using specific SSH profile"
    echo "  SSH_PROFILE=other-profile $0               # Using environment variable"
    echo "  $0 -h 54.123.45.67 -k ~/.ssh/my-key.pem    # Using explicit host and key"
    exit 1
}

# Parse command line arguments
NO_DEPLOY=false
EXPLICIT_HOST=false
EXPLICIT_KEY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--profile)
            SSH_PROFILE="$2"
            shift 2
            ;;
        -h|--host)
            EC2_HOST="$2"
            EXPLICIT_HOST=true
            shift 2
            ;;
        -u|--user)
            EC2_USER="$2"
            shift 2
            ;;
        -k|--key)
            EC2_KEY="$2"
            EXPLICIT_KEY=true
            shift 2
            ;;
        -d|--dir)
            REMOTE_DIR="$2"
            shift 2
            ;;
        --no-deploy)
            NO_DEPLOY=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# If host or key were explicitly provided, disable SSH profile mode
if [ "$EXPLICIT_HOST" = true ] || [ "$EXPLICIT_KEY" = true ]; then
    SSH_PROFILE=""
fi

# Validate required parameters
if [ -n "$SSH_PROFILE" ]; then
    # Using SSH profile mode
    USE_SSH_PROFILE=true
    # Test if SSH profile exists
    if ! ssh -G "$SSH_PROFILE" > /dev/null 2>&1; then
        print_error "SSH profile '$SSH_PROFILE' not found in ~/.ssh/config"
        exit 1
    fi
    # Extract host from SSH config for display purposes
    EC2_HOST=$(ssh -G "$SSH_PROFILE" | grep "^hostname " | awk '{print $2}')
    EC2_USER=$(ssh -G "$SSH_PROFILE" | grep "^user " | awk '{print $2}')
    if [ -z "$EC2_USER" ]; then
        EC2_USER="ubuntu"
    fi
else
    # Using traditional host/key mode
    USE_SSH_PROFILE=false
    if [ -z "$EC2_HOST" ]; then
        print_error "Either SSH_PROFILE or EC2_HOST is required"
        echo ""
        usage
    fi

    if [ -z "$EC2_KEY" ]; then
        print_error "EC2_KEY (SSH key path) is required when not using SSH profile"
        echo ""
        usage
    fi

    # Expand tilde in key path
    EC2_KEY="${EC2_KEY/#\~/$HOME}"

    # Check if key file exists
    if [ ! -f "$EC2_KEY" ]; then
        print_error "SSH key file not found: $EC2_KEY"
        exit 1
    fi

    # Make key file readable only by owner
    chmod 600 "$EC2_KEY" 2>/dev/null || true
fi

# Get project root directory (parent of script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

print_info "Project root: $PROJECT_ROOT"
if [ "$USE_SSH_PROFILE" = true ]; then
    print_info "SSH Profile: $SSH_PROFILE"
    print_info "EC2 Host: $EC2_HOST"
    print_info "EC2 User: $EC2_USER"
else
    print_info "EC2 Host: $EC2_HOST"
    print_info "EC2 User: $EC2_USER"
    print_info "SSH Key: $EC2_KEY"
fi
print_info "Remote Dir: $REMOTE_DIR"

# Build SSH command based on mode
if [ "$USE_SSH_PROFILE" = true ]; then
    SSH_CMD="ssh $SSH_PROFILE"
    RSYNC_CMD="rsync -avz --progress -e \"ssh\""
    SSH_DEST="$SSH_PROFILE"
else
    SSH_CMD="ssh -i $EC2_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    RSYNC_CMD="rsync -avz --progress -e \"ssh -i $EC2_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null\""
    SSH_DEST="$EC2_USER@$EC2_HOST"
fi

# Test SSH connection
print_info "Testing SSH connection..."
if ! $SSH_CMD "echo 'Connection successful'" > /dev/null 2>&1; then
    print_error "Failed to connect to EC2 instance"
    print_info "Please verify:"
    print_info "  - EC2 instance is running"
    print_info "  - Security group allows SSH from your IP"
    if [ "$USE_SSH_PROFILE" = false ]; then
        print_info "  - SSH key file path is correct"
        print_info "  - Username is correct (try 'ubuntu' or 'ec2-user')"
    else
        print_info "  - SSH profile '$SSH_PROFILE' is configured correctly"
    fi
    exit 1
fi
print_success "SSH connection successful"

# Check if docker-compose is installed on remote
print_info "Checking docker-compose installation..."
if ! $SSH_CMD "command -v docker-compose > /dev/null 2>&1"; then
    print_warning "docker-compose not found on remote. Installing..."
    $SSH_CMD << 'EOF'
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
EOF
    print_success "docker-compose installed"
else
    print_success "docker-compose is installed"
fi

# Create remote directory if it doesn't exist
print_info "Creating remote directory if needed..."
$SSH_CMD "mkdir -p $REMOTE_DIR"
print_success "Remote directory ready"

# Sync files (excluding unnecessary files)
print_info "Syncing files to EC2..."
eval $RSYNC_CMD \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '*.pyo' \
    --exclude '.pytest_cache' \
    --exclude 'htmlcov' \
    --exclude '.coverage' \
    --exclude '*.egg-info' \
    --exclude 'venv' \
    --exclude 'ENV' \
    --exclude 'env' \
    --exclude '.venv' \
    --exclude '.vscode' \
    --exclude '.idea' \
    --exclude '*.swp' \
    --exclude '*.swo' \
    --exclude '*~' \
    --exclude '.DS_Store' \
    --exclude 'Thumbs.db' \
    --exclude '*.log' \
    --exclude 'models/embedding_models/' \
    --exclude 'instructions/' \
    --exclude '.claude/' \
    --exclude 'debug_server.py' \
    --exclude 'src/simple_server.py' \
    --exclude 'src/notebooks/' \
    "$PROJECT_ROOT/" "$SSH_DEST:$REMOTE_DIR/"

print_success "Files synced successfully"

if [ "$NO_DEPLOY" = true ]; then
    print_info "Skipping deployment (--no-deploy flag set)"
    exit 0
fi

# Deploy using docker-compose
print_info "Deploying application on EC2..."
$SSH_CMD << EOF
    cd $REMOTE_DIR
    docker-compose down 2>/dev/null || true
    docker-compose up --build -d
EOF

print_success "Deployment initiated"

# Wait a moment for services to start
sleep 3

# Check deployment status
print_info "Checking deployment status..."
$SSH_CMD << EOF
    cd $REMOTE_DIR
    echo ""
    echo "=== Container Status ==="
    docker-compose ps
    echo ""
    echo "=== Recent Logs ==="
    docker-compose logs --tail=20
EOF

print_success "Deployment complete!"
print_info ""
print_info "Access your application:"
print_info "  Frontend UI:  http://$EC2_HOST:8000"
print_info "  Backend API:  http://$EC2_HOST:8080"
print_info ""
if [ "$USE_SSH_PROFILE" = true ]; then
    print_info "To view logs: ssh $SSH_PROFILE 'cd $REMOTE_DIR && docker-compose logs -f'"
    print_info "To check status: ssh $SSH_PROFILE 'cd $REMOTE_DIR && docker-compose ps'"
else
    print_info "To view logs: ssh -i $EC2_KEY $EC2_USER@$EC2_HOST 'cd $REMOTE_DIR && docker-compose logs -f'"
    print_info "To check status: ssh -i $EC2_KEY $EC2_USER@$EC2_HOST 'cd $REMOTE_DIR && docker-compose ps'"
fi


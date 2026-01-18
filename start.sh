#!/bin/bash

# ProductGPT Evaluation Tool - Startup Script

echo "================================================"
echo "  ProductGPT Evaluation Pipeline"
echo "  Starting Application..."
echo "================================================"
echo ""

# Check if Docker is installed
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "✓ Docker and Docker Compose detected"
    echo ""
    
    read -p "Deploy with Docker? (y/n): " use_docker
    
    if [ "$use_docker" = "y" ]; then
        echo ""
        echo "Building and starting Docker containers..."
        docker-compose up --build -d
        
        echo ""
        echo "================================================"
        echo "✓ Application started successfully!"
        echo "================================================"
        echo ""
        echo "Access the application at: http://localhost:8501"
        echo "Default password: covergo2024"
        echo ""
        echo "To stop the application:"
        echo "  docker-compose down"
        echo ""
        echo "To view logs:"
        echo "  docker-compose logs -f"
        echo "================================================"
        exit 0
    fi
fi

# Local Python deployment
echo ""
echo "Setting up local Python environment..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 is not installed. Please install Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create data directories
mkdir -p data/uploads data/results

echo ""
echo "================================================"
echo "✓ Setup complete!"
echo "================================================"
echo ""
echo "Starting Streamlit application..."
echo ""

# Run Streamlit
streamlit run frontend/streamlit_app.py

# Cleanup on exit
deactivate

#!/bin/bash

echo "Setting up Mental Health AI Demo..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/eeg/raw data/eeg/processed
mkdir -p data/audio/raw data/audio/processed
mkdir -p data/text/raw data/text/processed
mkdir -p models
mkdir -p results/eeg results/audio results/text results/fusion
mkdir -p visualizations

# Run the quick demo
echo "Running quick demo..."
python quick_demo.py

# Generate visualizations
echo "Generating visualizations..."
python generate_visualizations.py

echo "Demo completed successfully!"
echo "Results are available in the demo_results directory."
echo "Visualizations are available in the visualizations directory."
echo
echo "Press Enter to exit..."
read

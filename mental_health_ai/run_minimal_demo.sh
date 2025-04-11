#!/bin/bash

echo "Setting up Minimal Mental Health AI Demo..."

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

# Install minimal dependencies
echo "Installing minimal dependencies..."
pip install numpy matplotlib scikit-learn

# Run the minimal demo
echo "Running minimal demo..."
python minimal_demo.py

echo "Demo completed successfully!"
echo "Results are available in the minimal_results directory."
echo
echo "Press Enter to exit..."
read

#!/bin/bash

# Set environment variables for Python (optional)
export PYTHONPATH=/usr/local/lib/python3.8/dist-packages

# Ensure DISPLAY is set correctly for Raspberry Pi (if applicable)
export DISPLAY=:0

# Activate virtual environment

source /home/omotec/Desktop/arwanisef/arwanisefenv/bin/activate

# Check if the required Python dependencies are installed
echo "Checking for required dependencies..."



# Run the Python script
echo "Starting Streaemlit Python script..."
streamlit run /home/omotec/Desktop/arwanisef/final_arwan_rpi_testing_tailsense.py

# Deactivate the virtual environment
deactivate

#!/bin/bash
# build.sh

set -euo pipefail

python -m venv venv

# Activate virtual environment based on OS
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    source venv/bin/activate
elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "win32"* ]]; then
    source venv/Scripts/activate
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

echo "Installing required packages"
pip install -r requirements.txt

echo "Build Complete. Run the program using:"
echo "python main.py --input path/to/dataset --output path/to/output"

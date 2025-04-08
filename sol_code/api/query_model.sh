#!/bin/bash

# Default values
NODE=${1:-"localhost"}  # First argument or default to localhost
QUERY=${2:-""}          # Second argument (optional query)
TIMEOUT=${3:-90}        # Third argument (optional timeout in seconds)

# Determine the path to this script and the project
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Go to the API directory
cd "$SCRIPT_DIR"

# Check if we should use interactive mode
if [ -z "$QUERY" ]; then
    # No query provided, use interactive mode
    python client.py --interactive --node "$NODE" --timeout "$TIMEOUT"
else
    # Query provided, run once
    python client.py --query "$QUERY" --node "$NODE" --timeout "$TIMEOUT"
fi

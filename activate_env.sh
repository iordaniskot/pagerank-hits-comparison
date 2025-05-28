#!/bin/bash

# Script to activate the virtual environment for Graph Theory project
# Usage: source activate_env.sh

echo "Activating Graph Theory virtual environment..."
source graph_theory_env/bin/activate

echo "Virtual environment activated!"
echo "Python path: $(which python)"
echo "Installed packages:"
pip list --format=columns

echo ""
echo "To run the PageRank vs HITS comparison:"
echo "python run_complete_analysis.py"
echo ""
echo "To deactivate when done:"
echo "deactivate"

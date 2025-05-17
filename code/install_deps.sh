#!/bin/bash

# Shell script to install Python dependencies from requirements.txt

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo -e "\033[31mError: requirements.txt file not found!\033[0m"
    exit 1
fi

# Verify pip command availability
if ! command -v pip &> /dev/null; then
    echo -e "\033[31mError: pip command not found. Please install pip first.\033[0m"
    exit 1
fi

echo -e "\033[34mStarting dependency installation...\033[0m"

# Execute installation command
pip install -r requirements.txt

# Check installation result status
if [ $? -eq 0 ]; then
    echo -e "\033[32mDependencies installed successfully!\033[0m"
else
    echo -e "\033[31mDependency installation failed. Please check error messages.\033[0m"
    exit 1
fi


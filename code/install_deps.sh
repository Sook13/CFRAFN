#!/usr/bin/env bash

set -euo pipefail

# Install Python dependencies from requirements.txt in this script directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    echo -e "\033[31mError: requirements.txt not found at ${REQUIREMENTS_FILE}\033[0m"
    exit 1
fi

if ! command -v python >/dev/null 2>&1; then
    echo -e "\033[31mError: python command not found. Install Python first.\033[0m"
    exit 1
fi

echo -e "\033[34mInstalling dependencies from ${REQUIREMENTS_FILE}...\033[0m"
python -m pip install -r "${REQUIREMENTS_FILE}"
echo -e "\033[32mDependencies installed successfully.\033[0m"

#!/bin/bash

# Check if the folder is passed as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <folder>"
    exit 1
fi

# Assign the folder to a variable
FOLDER=$1

# Check if the folder exists
if [ ! -d "$FOLDER" ]; then
    echo "Error: $FOLDER is not a directory"
    exit 1
fi

# Extract the folder name and craft the output file name
FOLDER_NAME=$(basename "$FOLDER")
OUTPUT_FILE="${FOLDER_NAME}_python_dump.txt"

# Clear the output file if it already exists
> "$OUTPUT_FILE"

# Find Python files and append their content to the crafted output file
find "$FOLDER" -type f -name '*.py' | while read file; do
    echo "================== $file ==================" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo >> "$OUTPUT_FILE"
done

echo "Python code dump saved to $OUTPUT_FILE"

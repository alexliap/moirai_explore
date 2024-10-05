#!/bin/bash

# Define the URLs for the files to download
FILE_URL_1="https://eepublicdownloads.blob.core.windows.net/public-cdn-container/clean-documents/Publications/Statistics/2023/monthly_hourly_load_values_2023.csv"
FILE_URL_2="https://eepublicdownloads.blob.core.windows.net/public-cdn-container/clean-documents/Publications/Statistics/2024/monthly_hourly_load_values_2024.csv"

# Define the directory to save the files
DEST_DIR="data/"

# Check if the destination directory exists, if not, create it
if [ ! -d "$DEST_DIR" ]; then
  mkdir -p "$DEST_DIR"
fi

# Download the files using curl
curl -o "$DEST_DIR/file1.txt" "$FILE_URL_1"
curl -o "$DEST_DIR/file2.txt" "$FILE_URL_2"

# Inform the user that the download is complete
echo "Files have been downloaded to $DEST_DIR"

# Execute a Python command-line tool or script
echo "Running Python command..."
python -m uni2ts.data.builder.simple Load_Data data/load_data_23_24.csv --dataset_type wide --date_offset '2024-01-01 00:00:00'
# Inform the user that the Python command has been executed
echo "Python command executed successfully."

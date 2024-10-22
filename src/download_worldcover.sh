#!/bin/bash

BUCKET="esa-worldcover"
PREFIX="v200/2021/map"
FILES=()

# Get an initial batch of files
response=$(aws s3api list-objects --bucket "$BUCKET" --prefix "$PREFIX" --no-sign-request)
files_batch=($(echo "$response" | jq -r '.Contents[].Key'))
FILES=("${FILES[@]}" "${files_batch[@]}")

# Pagination loop
while $(echo "$response" | jq -e '.IsTruncated' &> /dev/null); do
    last_key="${files_batch[-1]}"
    response=$(aws s3api list-objects --bucket "$BUCKET" --prefix "$PREFIX" --no-sign-request --start-after "$last_key")
    files_batch=($(echo "$response" | jq -r '.Contents[].Key'))
    FILES=("${FILES[@]}" "${files_batch[@]}")
done

TOTAL_FILES=${#FILES[@]}
echo "Total files found: $TOTAL_FILES"

# Save the entire list of filenames to list_of_files.txt
printf "%s\n" "${FILES[@]}" > list_of_files.txt

# Ensure data directory exists
mkdir -p data

# Download all the files
printf "%s\n" "${FILES[@]}" | xargs -I{} -P30 sh -c 'mkdir -p "data/$(dirname "{}")"; aws s3 cp "s3://'$BUCKET'/{}" "data/{}" --no-sign-request'

echo "Filenames saved to list_of_files.txt and all files downloaded."

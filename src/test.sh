#!/bin/bash

BUCKET="esa-worldcover"
PREFIX="v200/2021/map"

# Ensure data directory exists
mkdir -p data

# Download all files under the prefix to the data directory
aws s3 cp "s3://$BUCKET/$PREFIX" data --recursive --no-sign-request

echo "All files downloaded to the data directory."

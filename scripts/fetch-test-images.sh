#!/bin/bash
set -e

RELEASE_TAG="v0.0.1-testdata"
REPO="kathishah/sundayalbum-claude"
ZIP_FILE="test-images.zip"
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${ZIP_FILE}"

echo "📥 Downloading test images from GitHub release ${RELEASE_TAG}..."
curl -L -o "${ZIP_FILE}" "${DOWNLOAD_URL}"

echo "📦 Extracting test images..."
unzip -o "${ZIP_FILE}"

echo "🧹 Cleaning up..."
rm "${ZIP_FILE}"

echo "✅ Downloaded $(ls test-images/ | wc -l) test images (~140MB total)"


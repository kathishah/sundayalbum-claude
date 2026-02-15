#!/bin/bash
gh release download v0.0.1-testdata --pattern "test-images.zip"
unzip -o test-images.zip
rm test-images.zip
echo "Downloaded $(ls test-images/ | wc -l) test images"


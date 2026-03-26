#!/bin/bash
# build-release.sh — Build a distributable Sunday Album .app and zip it.
#
# Usage:
#   cd mac-app && ./build-release.sh
#
# Output: ~/Desktop/SundayAlbum-<version>.zip
#
# NOTE: The .app is ad-hoc signed (no Apple Developer account required).
# Recipients must right-click → Open on first launch to clear Gatekeeper quarantine.
# For proper notarization, get an Apple Developer account and switch CODE_SIGN_IDENTITY
# to your "Developer ID Application: <Name>" cert in project.yml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_PATH="$SCRIPT_DIR/.build/SundayAlbum.xcarchive"
EXPORT_PATH="$SCRIPT_DIR/.build/export"
VERSION=$(grep MARKETING_VERSION "$SCRIPT_DIR/project.yml" | head -1 | awk -F'"' '{print $2}')
OUT_ZIP="$HOME/Desktop/SundayAlbum-${VERSION}.zip"

echo "==> Regenerating Xcode project..."
cd "$SCRIPT_DIR"
xcodegen generate

echo "==> Archiving (Release)..."
xcodebuild -scheme SundayAlbum \
           -configuration Release \
           -archivePath "$ARCHIVE_PATH" \
           archive \
           | grep -E "error:|warning:|Archive Succeeded|Build Succeeded" || true

echo "==> Exporting .app..."
xcodebuild -exportArchive \
           -archivePath "$ARCHIVE_PATH" \
           -exportOptionsPlist "$SCRIPT_DIR/ExportOptions.plist" \
           -exportPath "$EXPORT_PATH" \
           | grep -E "error:|Export Succeeded" || true

APP_PATH="$EXPORT_PATH/SundayAlbum.app"
if [ ! -d "$APP_PATH" ]; then
    echo "✗ Export failed — .app not found at $APP_PATH"
    exit 1
fi

echo "==> Zipping to Desktop..."
cd "$EXPORT_PATH"
zip -r "$OUT_ZIP" "SundayAlbum.app"

echo ""
echo "✓ Done: $OUT_ZIP"
echo ""
echo "Share this zip with friends. They should:"
echo "  1. Unzip it"
echo "  2. Drag SundayAlbum.app to /Applications"
echo "  3. Right-click → Open on first launch (bypasses Gatekeeper for unsigned apps)"
echo "  4. The app will install its Python environment automatically (~2 min, once)"

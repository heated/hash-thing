#!/usr/bin/env bash
# notarize-macos-app.sh — build a minimal .app bundle around the release binary,
# sign it with a Developer ID Application identity, notarize the zip archive,
# staple the ticket to the app bundle, then re-zip the stapled app for release.
#
# Apple requires uploading a zip/dmg/pkg to the notary service, and their
# notarization docs note that zip files can't be stapled directly — staple the
# app bundle first, then create a fresh zip for distribution.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <release-binary> <output-dir>" >&2
    exit 1
fi

BIN_PATH="$1"
OUT_DIR="$2"

APP_NAME="${MACOS_APP_NAME:-hash-thing}"
BUNDLE_ID="${MACOS_BUNDLE_ID:-io.github.heated.hash-thing}"
APP_VERSION="${MACOS_APP_VERSION:-0.0.0}"
ARCHIVE_NAME="${MACOS_ARCHIVE_NAME:-hash-thing-macos-aarch64.zip}"
APP_VERSION="${APP_VERSION#v}"

: "${MACOS_CODESIGN_IDENTITY:?MACOS_CODESIGN_IDENTITY is required}"
: "${MACOS_NOTARY_PROFILE:?MACOS_NOTARY_PROFILE is required}"
: "${MACOS_KEYCHAIN_PATH:?MACOS_KEYCHAIN_PATH is required}"

if [[ ! -f "$BIN_PATH" ]]; then
    echo "error: release binary not found at $BIN_PATH" >&2
    exit 1
fi

APP_DIR="$OUT_DIR/$APP_NAME.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
ARCHIVE_PATH="$OUT_DIR/$ARCHIVE_NAME"

rm -rf "$APP_DIR"
mkdir -p "$OUT_DIR"
mkdir -p "$MACOS_DIR"
cp "$BIN_PATH" "$MACOS_DIR/$APP_NAME"
chmod +x "$MACOS_DIR/$APP_NAME"

cat > "$CONTENTS_DIR/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>${APP_VERSION}</string>
    <key>CFBundleVersion</key>
    <string>${APP_VERSION}</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
</dict>
</plist>
EOF

plutil -lint "$CONTENTS_DIR/Info.plist"

codesign \
    --force \
    --options runtime \
    --timestamp \
    --sign "$MACOS_CODESIGN_IDENTITY" \
    "$APP_DIR"

codesign --verify --deep --strict --verbose=2 "$APP_DIR"

rm -f "$ARCHIVE_PATH"
/usr/bin/ditto -c -k --keepParent "$APP_DIR" "$ARCHIVE_PATH"

xcrun notarytool submit \
    "$ARCHIVE_PATH" \
    --keychain-profile "$MACOS_NOTARY_PROFILE" \
    --keychain "$MACOS_KEYCHAIN_PATH" \
    --wait

xcrun stapler staple "$APP_DIR"
xcrun stapler validate "$APP_DIR"

rm -f "$ARCHIVE_PATH"
/usr/bin/ditto -c -k --keepParent "$APP_DIR" "$ARCHIVE_PATH"

codesign --verify --deep --strict --verbose=2 "$APP_DIR"
spctl -a -t exec -vv "$APP_DIR"

echo "notarized archive ready: $ARCHIVE_PATH"

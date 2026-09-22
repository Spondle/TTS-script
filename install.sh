#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p ~/.local/bin
ln -sf "$PROJECT_DIR/main.py" ~/.local/bin/tts
chmod +x "$PROJECT_DIR/main.py"

echo "Successfully installed 'tts' to ~/.local/bin/tts"
echo "You can now run 'tts' from anywhere in your terminal!"

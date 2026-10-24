#!/bin/bash
# update.sh  -  installs the newest lockit_update zip from your Downloads folder.
#   bash update.sh
# Your data folder and .env (API key) are never touched.

DIR="$(cd "$(dirname "$0")" && pwd)"
ZIP=$(ls -t ~/Downloads/lockit_update*.zip 2>/dev/null | head -1)

if [ -z "$ZIP" ]; then
  echo "No lockit_update zip found in Downloads. Download it from the chat first."
  exit 1
fi

echo "Installing: $ZIP"
unzip -o -q "$ZIP" -d "$DIR" -x "update.sh"
"$DIR/.venv/bin/pip" install -q -r "$DIR/requirements.txt" 2>/dev/null
mv "$ZIP" ~/.Trash/ 2>/dev/null   # so next time the newest zip is the only one

echo "Done. Start the dashboard with:  streamlit run lockit_app.py"

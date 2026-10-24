#!/bin/bash
# setup_automation.sh  -  schedules auto.py to run on its own.
#   bash setup_automation.sh          turn automation on (safe to run again)
#   bash setup_automation.sh off      turn automation off

DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$DIR/.venv/bin/python3"
AGENTS="$HOME/Library/LaunchAgents"
LOGS="$DIR/data/logs"
JOBS="morning snapshots"

for job in $JOBS; do
  launchctl unload "$AGENTS/com.lockitlabs.$job.plist" 2>/dev/null
done

if [ "$1" = "off" ]; then
  for job in $JOBS; do rm -f "$AGENTS/com.lockitlabs.$job.plist"; done
  echo "Automation is OFF."
  exit 0
fi

case "$DIR" in
  "$HOME/Desktop"*|"$HOME/Documents"*|"$HOME/Downloads"*)
    echo "STOP: this folder is in Desktop, Documents, or Downloads."
    echo "macOS blocks scheduled jobs from reading those folders."
    echo "Move the project to your home folder first (see the instructions in chat)."
    exit 1 ;;
esac

if [ ! -x "$PY" ]; then
  echo "STOP: no virtual environment found at $PY"
  echo "Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

mkdir -p "$AGENTS" "$LOGS"

write_plist() {  # $1 = job name, $2 = schedule XML
  cat > "$AGENTS/com.lockitlabs.$1.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.lockitlabs.$1</string>
  <key>ProgramArguments</key>
  <array><string>$PY</string><string>$DIR/auto.py</string><string>$1</string></array>
  <key>WorkingDirectory</key><string>$DIR</string>
  $2
  <key>StandardOutPath</key><string>$LOGS/$1.log</string>
  <key>StandardErrorPath</key><string>$LOGS/$1.log</string>
</dict>
</plist>
EOF
  launchctl load "$AGENTS/com.lockitlabs.$1.plist"
}

write_plist morning '<key>StartCalendarInterval</key><dict><key>Hour</key><integer>9</integer><key>Minute</key><integer>0</integer></dict>'
write_plist snapshots '<key>StartInterval</key><integer>900</integer>'

echo "Automation is ON."
echo "  morning:   box scores + grading every day at 9:00 AM"
echo "  snapshots: checks every 15 minutes for games about to tip off"
echo "Logs: $LOGS"
launchctl list | grep lockitlabs

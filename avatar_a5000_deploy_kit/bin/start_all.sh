#!/usr/bin/env bash
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ROOT="$(cd "$KIT/../avatar_a5000/realtime" && pwd -P)"
NGROK=/opt/avatar_tools/ngrok
LOGDIR=/opt/avatar_logs

mkdir -p "$LOGDIR"

ready() {
    curl -fsS --max-time 3 "$1" >/dev/null 2>&1
}

echo "============================================"
echo "    STARTING COMPLETE AVATAR SYSTEM"
echo "============================================"

# --------------------------------------------
# STEP 1: WHISPER
# --------------------------------------------

echo
echo "[1/3] WHISPER STT — GPU 1"

if ready "http://127.0.0.1:8310/docs"; then
    echo "[OK] Whisper is already ready."
    WHISPER_READY=1

elif pgrep -f 'scripts/stt_server[.]py' >/dev/null; then
    echo "[INFO] Whisper is already loading."
    echo "[INFO] Reusing the existing process."
    WHISPER_READY=0

else
    echo "[START] Starting Whisper in background..."

    nohup env \
        AVATAR_ROOT="$ROOT" \
        KIT="$KIT" \
        STT_GPU=1 \
        STT_PORT=8310 \
        bash "$KIT/bin/07_start_local_whisper.sh" \
        > "$LOGDIR/whisper.log" 2>&1 < /dev/null &

    echo "[OK] Whisper launcher PID: $!"
    WHISPER_READY=0
fi

# --------------------------------------------
# STEP 2: WAIT 60 SECONDS
# --------------------------------------------

if [ "$WHISPER_READY" -eq 0 ]; then
    echo
    echo "[WAIT] Giving Whisper 60 seconds to load..."

    for i in 1 2 3 4 5 6; do
        sleep 10
        echo "[WAIT] $((i * 10))/60 seconds"

        if ready "http://127.0.0.1:8310/docs"; then
            echo "[OK] Whisper became ready."
            break
        fi
    done

    if ! ready "http://127.0.0.1:8310/docs"; then
        echo "[INFO] Whisper is still loading."
        echo "[INFO] Continuing to MuseTalk as requested."
    fi
fi

# --------------------------------------------
# STEP 3: MUSETALK AVATAR
# --------------------------------------------

echo
echo "[2/3] MUSETALK AVATAR — GPU 0"

if ready "http://127.0.0.1:8782/"; then
    echo "[OK] MuseTalk is already ready."

elif pgrep -f 'avatar_system/streaming/server[.]py' >/dev/null; then
    echo "[INFO] MuseTalk process already exists."
    echo "[INFO] Not starting a duplicate."

else
    echo "[START] Starting MuseTalk in background..."

    nohup env -u SARVAM_API_KEY \
        AVATAR_ROOT="$ROOT" \
        KIT="$KIT" \
        PATH="/opt/avatar_envs/musetalk/bin:$PATH" \
        GPU=0 \
        STT_MODE=local \
        STT_URL=http://127.0.0.1:8310 \
        PORT=8782 \
        bash "$KIT/bin/08_start_avatar.sh" \
        > "$LOGDIR/avatar.log" 2>&1 < /dev/null &

    echo "[OK] MuseTalk launcher PID: $!"
fi

# Small delay before starting ngrok.
sleep 5

# --------------------------------------------
# STEP 4: NGROK
# --------------------------------------------

echo
echo "[3/3] NGROK HTTPS TUNNEL"

if [ ! -x "$NGROK" ]; then
    echo "[WARN] ngrok executable not found: $NGROK"

elif pgrep -x ngrok >/dev/null; then
    echo "[OK] An ngrok process is already running."

else
    echo "[START] Starting ngrok in background..."

    nohup "$NGROK" http 127.0.0.1:8782 \
        > "$LOGDIR/ngrok.log" 2>&1 < /dev/null &

    echo "[OK] ngrok launcher PID: $!"
fi

sleep 5

# --------------------------------------------
# FINAL STATUS
# --------------------------------------------

echo
echo "============================================"
echo "             SERVER STATUS"
echo "============================================"

if ready "http://127.0.0.1:8310/docs"; then
    echo "[READY] Whisper STT :8310"
else
    echo "[LOADING] Whisper STT :8310"
fi

if ready "http://127.0.0.1:8782/"; then
    echo "[READY] MuseTalk avatar :8782"
else
    echo "[LOADING] MuseTalk avatar :8782"
fi

URL=$(
    curl -fsS --max-time 3 \
        http://127.0.0.1:4040/api/tunnels 2>/dev/null \
    | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
    for tunnel in data.get("tunnels", []):
        url = tunnel.get("public_url", "")
        addr = str(tunnel.get("config", {}).get("addr", ""))
        if url.startswith("https://") and (
            addr.endswith(":8782") or addr.endswith("localhost:8782")
        ):
            print(url)
            break
except Exception:
    pass
' 2>/dev/null || true
)

if [ -n "$URL" ]; then
    echo
    echo "AVATAR WEBSITE:"
    echo "$URL"
else
    echo "[INFO] ngrok URL not confirmed yet."
    echo "Check $LOGDIR/ngrok.log"
fi

echo
echo "============================================"
echo "          START COMMAND FINISHED"
echo "============================================"
echo
echo "Services continue running in background."
echo "Whisper log: $LOGDIR/whisper.log"
echo "Avatar log:  $LOGDIR/avatar.log"
echo "ngrok log:   $LOGDIR/ngrok.log"


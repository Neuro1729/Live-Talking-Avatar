#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "      STOPPING COMPLETE AVATAR SYSTEM"
echo "========================================"

python3 - <<'PY'
import os
import signal
import time

def processes():
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == os.getpid():
            continue
        try:
            data = open(
                f"/proc/{pid}/cmdline", "rb"
            ).read()
            args = [
                x.decode(errors="replace")
                for x in data.split(b"\0") if x
            ]
            if args:
                found.append((pid, args))
        except (OSError, ProcessLookupError):
            pass
    return found

def matches(args, service):
    if service == "ngrok":
        return (
            args[0].startswith("/opt/avatar_tools/ngrok")
            and "http" in args
            and any("8782" in a for a in args)
        )

    if service == "avatar":
        return (
            any(a.endswith("avatar_system/streaming/server.py")
                for a in args)
            and "--port" in args
            and "8782" in args
        ) or any(
            a.endswith("/bin/08_start_avatar.sh")
            for a in args
        )

    if service == "whisper":
        return (
            any(a.endswith("scripts/stt_server.py")
                for a in args)
            and "--port" in args
            and "8310" in args
        ) or any(
            a.endswith("/bin/07_start_local_whisper.sh")
            for a in args
        )

    return False

def find(service):
    return [
        pid for pid, args in processes()
        if matches(args, service)
    ]

def stop(service, label):
    print(f"\n=== Stopping {label} ===", flush=True)
    pids = find(service)

    if not pids:
        print(f"[OK] {label} is already stopped.")
        return

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[TERM] {label}: PID {pid}")
        except ProcessLookupError:
            pass

    for _ in range(30):
        if not find(service):
            print(f"[OK] {label} stopped.")
            return
        time.sleep(0.5)

    remaining = find(service)

    if remaining:
        print(f"[WARN] {label} did not exit gracefully.")
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"[KILL] {label}: PID {pid}")
            except ProcessLookupError:
                pass

    time.sleep(0.5)

    if find(service):
        print(f"[WARN] {label} still has processes.")
    else:
        print(f"[OK] {label} stopped.")

stop("ngrok", "ngrok tunnel")
stop("avatar", "MuseTalk avatar")
stop("whisper", "Whisper STT")

print("\n========================================")
print("        AVATAR SHUTDOWN FINISHED")
print("========================================")
PY

echo
echo "=== CHECK PORTS ==="

for port in 8782 8310; do
    if curl -fsS --max-time 2 \
        "http://127.0.0.1:$port/" \
        >/dev/null 2>&1; then
        echo "[WARN] Port $port still responds."
    else
        echo "[OK] Port $port is not responding on /."
    fi
done

echo
echo "Docker and vLLM were not stopped."

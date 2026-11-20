"""Explicit behaviour state machine: LISTENING / THINKING / SPEAKING / INTERRUPTED / RECOVERY.

Events (from the conversation engine and the motion scheduler):
  user_speaking   -> LISTENING          user_done   -> THINKING        audio_ready -> SPEAKING
  response_done   -> LISTENING          interrupt   -> INTERRUPTED     cancelled   -> RECOVERY
  hub_reached (scheduler, motion back at neutral) -> LISTENING (from RECOVERY/INTERRUPTED)
Invalid events are ignored and counted (never raise inside the real-time loop).
"""
import threading
import time

STATES = ("LISTENING", "THINKING", "SPEAKING", "INTERRUPTED", "RECOVERY")
TRANSITIONS = {
    ("LISTENING", "user_speaking"): "LISTENING", ("THINKING", "user_speaking"): "LISTENING",
    ("RECOVERY", "user_speaking"): "LISTENING",
    ("LISTENING", "user_done"): "THINKING", ("RECOVERY", "user_done"): "THINKING",
    ("THINKING", "audio_ready"): "SPEAKING", ("LISTENING", "audio_ready"): "SPEAKING",
    ("RECOVERY", "audio_ready"): "SPEAKING", ("SPEAKING", "audio_ready"): "SPEAKING",
    ("SPEAKING", "response_done"): "LISTENING", ("THINKING", "response_done"): "LISTENING",
    ("SPEAKING", "interrupt"): "INTERRUPTED", ("THINKING", "interrupt"): "INTERRUPTED",
    ("LISTENING", "interrupt"): "LISTENING",
    ("INTERRUPTED", "cancelled"): "RECOVERY",
    ("RECOVERY", "hub_reached"): "LISTENING", ("INTERRUPTED", "hub_reached"): "RECOVERY",
    ("SPEAKING", "user_speaking"): "INTERRUPTED",
}


class StateMachine:
    def __init__(self, on_change=None):
        self.state, self.since = "LISTENING", time.time()
        self.lock = threading.Lock()
        self.history = []
        self.ignored = 0
        self.on_change = on_change

    def fire(self, event):
        with self.lock:
            new = TRANSITIONS.get((self.state, event))
            if new is None:
                self.ignored += 1
                return self.state
            old = self.state
            if new != old:
                self.state, self.since = new, time.time()
                self.history.append((round(self.since, 3), old, event, new))
                del self.history[:-200]
        if new != old and self.on_change:
            self.on_change(old, new, event)
        return new

    def snapshot(self):
        return {"state": self.state, "for_s": round(time.time() - self.since, 2), "ignored_events": self.ignored,
                "recent": self.history[-8:]}

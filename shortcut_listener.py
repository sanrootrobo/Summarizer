# shortcut_listener.py

import subprocess
import sys
from pynput import keyboard

# Define the hotkey combination
# For macOS, this is Cmd+Y
# For Windows/Linux, we'll use Ctrl+Alt+Y
HOTKEY = [
    {keyboard.Key.cmd, keyboard.KeyCode(char='y')},
    {keyboard.Key.ctrl, keyboard.Key.alt, keyboard.KeyCode(char='y')}
]

# The script to launch when the hotkey is pressed
APP_SCRIPT = "spotlight_app.py"

def on_activate():
    print("Hotkey activated! Launching the spotlight app...")
    # Use Popen to launch the app as a separate, non-blocking process
    # We use sys.executable to ensure it runs with the same python interpreter
    subprocess.Popen([sys.executable, APP_SCRIPT])

def for_canonical(f):
    """A decorator to handle key normalization for the listener."""
    return lambda k: f(listener.canonical(k))

# Create a set of hotkeys to listen for
hotkeys = {
    tuple(sorted(key_set, key=lambda k: str(k))) for key_set in HOTKEY
}
current_keys = set()

def on_press(key):
    # Use canonical representation to handle different key objects
    if listener.canonical(key) in {k for key_set in hotkeys for k in key_set}:
        current_keys.add(listener.canonical(key))
        
        # Check if any of the hotkey combinations are met
        for key_set in hotkeys:
            if key_set.issubset(current_keys):
                on_activate()
                # Optional: break if you only want one hotkey to fire at a time
                break 

def on_release(key):
    try:
        current_keys.remove(listener.canonical(key))
    except KeyError:
        pass

print("Shortcut listener started.")
print("Press Cmd+Y (macOS) or Ctrl+Alt+Y (Windows/Linux) to launch the app.")
print("Press Ctrl+C in this terminal to stop the listener.")

# Collect events until released
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    try:
        listener.join()
    except KeyboardInterrupt:
        print("Listener stopped.")
        listener.stop()

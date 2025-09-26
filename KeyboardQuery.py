"""
Key-to-Value Mapper with Toggle (pynput)
----------------------------------------
- Listens for global key *releases*.
- If released key's character is in dict D, prints the mapped value and calls a placeholder function.
- Backtick (`) toggles the active state ON/OFF.
- ESC exits the listener.
"""

from typing import Dict, Optional
from pynput import keyboard

def placeholder_function(value: str) -> str:
    """Placeholder for your action. Replace with your real function later."""
    # TODO: implement your actual action here
    print(f"(Function would run here with: {value})")

def listen_for_key(D: Dict[str, str]) -> str:
    """
    Start a global key listener that:
      - Acts only on key release.
      - Toggles activation with backtick (` or ~).
      - Exits on ESC.
      - If active and key is a single character present in D, prints & runs placeholder_function.

    Parameters
    ----------
    D : dict[str, str]
        Dictionary mapping single-character keys (e.g., 'a', 'b', 'c') to strings.
    """
    state = {"active": True}  # mutable container for closure

    print("Listening for key releases... (` toggles ON/OFF, ESC quits)")

    def on_release(key: keyboard.Key) -> Optional[bool]:
        """Handle key release events only."""
        try:
            # Exit on ESC
            if key == keyboard.Key.esc:
                print("Exiting...")
                return False  # stop listener

            # Character keys
            if hasattr(key, "char") and key.char is not None:
                ch = key.char  # raw character, respects shift (e.g., '~')
                # Toggle on backtick or tilde (shifted backtick)
                if ch in ("`", "~"):
                    state["active"] = not state["active"]
                    print(f"Activation toggled: {'ON' if state['active'] else 'OFF'}")
                    return True

                if state["active"]:
                    # Normalize to lowercase to match dict keys like 'a'
                    k = ch.lower()
                    if len(k) == 1 and k in D:
                        value = D[k]
                        print(f"Key '{k}' → Value: {value}")
                        return placeholder_function(value) + value

        except Exception as e:
            # Keep listener alive even if something goes wrong
            print(f"(listener error: {e})")

        # Keep listening
        return True

    # Start listener (blocks until ESC)
    with keyboard.Listener(on_release=on_release) as listener:
        listener.join()

if __name__ == "__main__":
    # Example dictionary
    D = {"a": "You pressed A!", "b": "This is B’s value.", "c": "Mapped to C."}
    listen_for_key(D)

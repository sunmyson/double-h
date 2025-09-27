import keyboard

def wait_for_question_mark():
    """Block until the user presses the '?' key."""
    print("Press '?' to transcribe the last 30 seconds of system audio...")
    keyboard.wait("?")
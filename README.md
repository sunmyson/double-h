# double-h


__Summary:__

This Program takes the audio output from the computer, passes it through a Speech to Text Converter, and provides it as a prompt to an LLM, which prints its response onto a html server. At the end of the session, the logs of all prompts and responses are printed onto a .txt file

___Requirements:__

use terminal to:
install keyboard
install sounddevice as sd
from pprint install pprint
install io
install sys
install wave
install threading
install tempfile
from typing install Optional, Tuple
install numpy as np
install sounddevice as sd
from faster_whisper install WhisperModel
install time
install os
install tkinter as tk  
from tkinter install scrolledtext  



__USING LLM:__

_________________________________________

Please install the Ollama LLM (for macOS, enter "brew install ollama" into the terminal, Windows must download it from the website in the following way)

For Windows:
Installation Steps
1) Download the Installer
Go to the official Windows download page:
https://ollama.com/download/windows
Click the .exe installer.
2) Run the Installer
Double-click the OllamaSetup.exe file and follow the installation prompts.
By default, it installs under your user account (no elevated permissions needed).

Then, in a separate terminal, run 'ollama serve' (This terminal should not be touched for the running for the program)
_________________________________________

__Extra Steps for MacOS__

if on MacOS:
1) download and install BlackHole, and restart computer
2) go to Audio MIDI setup 
3) in the Bottom Left corner press the '+' icon and 'Create Multi-output Device'
4) Then, select BlackHole and Speakers
5) Then, right click on the new output device, and press "Use this device for sound output"

__How to use__ 
(assuming installations and Ollama steps have been completed)
1) In a new terminal, enter "python OverallSystem.py"
2) Open http://localhost:8000/LLM.html in a web browser (initially it wont exist)
3) Wait for "Press Enter to transcribe the last 30 seconds of system audio...Serving HTTP on :: port 8000 (http://[::]:8000/) ..." message.
4) Press enter into the terminal that you entered "python OverallSystem.py" to run it through the LLM, and after approx 10 secs reload the html page to see the prompt and LLM answer. 
5) Press ctrl+C to cancel the operation, and a log of all the previous conversations will popup for review




from zero_to_three import transcribe_last_30s
from llm import prompt_ollama_generate
import time
import os
import tkinter as tk  # NEW
from tkinter import scrolledtext  # NEW

# File to store the LLM responses
OUTPUT_HTML_PATH = "/Users/christoslellis/Downloads/double-h-1/LLM.html"
LOG_TXT_PATH = "/Users/christoslellis/Downloads/double-h-1/LLM_log.txt"

def update_html_file(file_path: str, prompt: str, response: str):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLM Output</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .prompt {{ font-weight: bold; color: #333; }}
            .response {{ margin-top: 10px; color: #555; }}
        </style>
    </head>
    <body>
        <h1>LLM Output</h1>
        <div class="prompt">Prompt:</div>
        <div>{prompt}</div>
        <div class="response">Response:</div>
        <div>{response}</div>
    </body>
    </html>
    """
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(html_content)

def append_text_log(file_path: str, prompt: str, response: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n--- Transcribing ---\n")
        f.write(f"Prompt: {prompt}\n\n")
        f.write("--- LLM Response ---\n")
        f.write(f"{response}\n")
        f.write("-" * 40 + "\n")

def start_http_server(directory: str, port: int = 8000):
    os.chdir(directory)
    os.system(f"python3 -m http.server {port}")

def show_popup(log_path: str):  # NEW
    root = tk.Tk()
    root.title("LLM_log.txt")

    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=30)
    text_area.pack(expand=True, fill="both")

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        content = "Log file not found."
    text_area.insert(tk.END, content)
    text_area.config(state=tk.DISABLED)  # make read-only

    root.mainloop()

if __name__ == "__main__":
    try:
        os.system('pkill -f "http.server.*8000" 2>/dev/null || true')
        directory = os.path.dirname(OUTPUT_HTML_PATH)
        os.system(f"python3 -m http.server --directory {directory} 8000 &")
        print(f"Serving HTML at http://localhost:8000/LLM.html")

        while True:
            input("Press Enter to transcribe the last 30 seconds of system audio...")
            print("\n--- Transcribing ---")
            prompt = transcribe_last_30s()
            print(f"Prompt: {prompt}")

            print("\n--- Fetching LLM Response ---")
            llm_response = prompt_ollama_generate("explain this " + prompt)
            print(f"LLM Response: {llm_response}")

            update_html_file(OUTPUT_HTML_PATH, prompt, llm_response)
            append_text_log(LOG_TXT_PATH, prompt, llm_response)

            time.sleep(1)

    except KeyboardInterrupt:
        print("Exiting...")
        show_popup(LOG_TXT_PATH)  # NEW

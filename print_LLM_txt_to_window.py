import tkinter as tk

def show_text_window(file_path: str):
    """
    Display the content of a text file in a popup window.

    Parameters
    ----------
    file_path : str
        Path to the text file to display.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text_content = file.read()
    except FileNotFoundError:
        text_content = "(File not found. Please ensure the file exists.)"
    except Exception as e:
        text_content = f"(Error reading file: {e})"

    # Create a simple Tkinter window
    root = tk.Tk()
    root.title("LLM Output")

    # Create a text widget to display the content
    text_area = tk.Text(root, wrap=tk.WORD, height=30, width=80)
    text_area.pack(expand=True, fill=tk.BOTH)

    # Insert the content into the text widget
    text_area.insert(tk.END, text_content + "\n\n")

    # Make the text widget read-only
    text_area.config(state=tk.DISABLED)

    # Run the Tkinter event loop
    root.mainloop()
from transformers import pipeline

# 1. Load a free, lightweight summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(input_text: str) -> str:
    """
    Summarize input text using a lightweight LLM.
    Args:
        input_text (str): The input transcript or text.
    Returns:
        str: A summary string.
    """
    # 2. Run summarization
    summary = summarizer(input_text, max_length=50, min_length=20, do_sample=False)
    
    # 3. Extract summary string
    return summary[0]['summary_text']


# Example usage:
if __name__ == "__main__":
    input_string = """
    In this video, the speaker talks about the importance of learning Python for data science.
    He explains how libraries like pandas and NumPy make data manipulation easier.
    He also touches on visualization tools like Matplotlib and Seaborn,
    and concludes by encouraging viewers to practice coding daily to build confidence.
    """
    
    output_string = summarize_text(input_string)
    print("===== Summary =====")
    print(output_string)

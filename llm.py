import httpx

def prompt_ollama_generate(prompt: str,
                           model: str = "phi3:mini",
                           base_url: str = "http://127.0.0.1:11434") -> str:
    payload = {
        "model": model,
        "prompt": f"Be concise and fast.\n\nUser: {prompt}\nAssistant:",
        "stream": False
    }
    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{base_url}/api/generate", json=payload)
        r.raise_for_status()
        j = r.json()
        return (j.get("response") or "").strip()

if __name__ == "__main__":
    print(prompt_ollama_generate("What is the capital of Greece?"))



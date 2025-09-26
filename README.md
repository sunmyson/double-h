# double-h
two gentle


Workflow:

[System Audio] --(loopback)--> [RingBuffer 30s]
                                  |
Hotkey '?' ----> [Freeze last 30s PCM] --> [ASR (faster-whisper)]
                                               |
                                 [transcript + timestamps]
                                               |
                                 [Task Router: summarize | explain | factcheck]
                                               |
                                    [LLM (lite) prompt templates]
                                               |
                            [Side Panel UI + Optional SQLite log]

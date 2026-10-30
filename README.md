# Real-Time Fact Checker (Speaker Diarization + Live Web UI)

This project listens to a live conversation through your microphone, transcribes it with speaker diarization (so it can tell speakers apart), and fact-checks the parts that sound like verifiable claims. While it runs, it streams the transcript and fact-check results to a web UI. When the session ends, it generates a short summary, a brief analysis of each speaker, and a basic trustworthiness score based on how accurate their claims were.

## What this project does
- Records live audio from your microphone  
- Transcribes speech with speaker diarization (Speaker 1, Speaker 2, etc.)  
- Classifies each spoken segment as:
  - **RESEARCH** (likely a factual claim)
  - **IGNORE** (opinions, small talk, filler, or personal statements)
- Fact-checks RESEARCH segments using Perplexity Sonar  
- Streams live updates to the browser using WebSockets:
  - transcript segments  
  - fact-check status and results  
- Generates a final report when recording stops:
  - 2–3 sentence conversation summary  
  - 1–2 sentence analysis per speaker  
  - trustworthiness score per speaker (based on TRUE / FALSE / PARTIAL verdicts)

## How it works (high level)
1. Audio is captured in real time using `sounddevice`.
2. Audio chunks are sent to ElevenLabs Scribe v2 for transcription with diarization.
3. Transcribed words are grouped by speaker into readable segments.
4. Each segment is classified by OpenAI as RESEARCH or IGNORE.
5. RESEARCH segments are queued to background workers that fact-check them via Perplexity Sonar.
6. Transcripts and verdicts are streamed to the frontend through a FastAPI WebSocket.
7. When the session ends, OpenAI generates a summary and speaker analysis, and the app computes trust scores.

## Setup

### Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt




## **Environment variables**
Create a .env file in the project root (do not commit this file):

env
ELEVENLABS_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
PPLX_API_KEY=your_key_here





## **Run the web app**
python app.py

### **Open:**
http://localhost:8000/
Use the UI to start and stop recording. When recording ends, the final analysis is automatically generated and sent to the UI.






## **Other entry points**
python fact_checker.py — terminal-based version (prints transcript and verdicts)

python transcribe.py — transcription-only mode (useful for testing audio and diarization)






## **Libraries used**
### **Standard library**
- os, sys, json, time, signal
- asyncio, threading
- io, wave
- queue (Queue, Empty)
- concurrent.futures (ThreadPoolExecutor)
- contextlib (asynccontextmanager)
- typing (List, Dict)
- pprint

### **Third-party libraries**
- fastapi (FastAPI, WebSocket, WebSocketDisconnect)
- fastapi.staticfiles (StaticFiles)
- fastapi.responses (HTMLResponse)
- uvicorn
- numpy
- sounddevice
- python-dotenv
- elevenlabs
- openai
- requests
- warnings





## **Notes**
Make sure microphone permissions are enabled for your terminal or IDE.

API keys should never be hardcoded or committed to GitHub.

Diarization quality depends heavily on audio quality and background noise.
# NexHacks_Project

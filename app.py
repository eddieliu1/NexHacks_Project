import os
import sys
import asyncio
import json
import threading
import time
from typing import List, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
import sounddevice as sd
from contextlib import asynccontextmanager

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import backend logic
from fact_checker import FactCheckTranscriber, FactCheckWorker, Colors, SAMPLE_RATE, CHANNELS, TRANSCRIPTION_INTERVAL, openai_client
from classifier import classify_statement

def force_statement(text: str) -> str:
    """Convert any text to a statement by removing question marks."""
    if text.endswith('?'):
        return text.rstrip('?').strip() + '.'
    return text

# Global state
transcriber = None

# Global state for connected clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

# Custom Worker that handles segment IDs for the web UI
class WebFactCheckWorker(FactCheckWorker):
    def __init__(self, speaker_id: str, result_callback):
        super().__init__(speaker_id, result_callback)
    
    def add_claim(self, text: str, segment_id: str):
        """Add a claim with its UI segment ID."""
        self.queue.put((text, segment_id))
    
    def _run(self):
        """Worker loop - processes claims from queue."""
        while self.running:
            try:
                # Expecting a tuple of (claim, segment_id)
                item = self.queue.get(timeout=0.5)
                claim, segment_id = item
                result = self._fact_check(claim)
                # Pass segment_id back to callback
                self.result_callback(self.speaker_id, claim, result, segment_id)
            except Exception:
                continue

# Custom Transcriber for the Web
class WebTranscriber(FactCheckTranscriber):
    def __init__(self, manager: ConnectionManager, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.manager = manager
        self.loop = loop
        self.segment_counter = 0
        self.recording_active = False
        self.speaker_fact_checks = {} # speaker_id -> list of verdicts
        self.transcript_for_analysis = [] # list of (speaker_id, text)

    def get_or_create_worker(self, speaker_id) -> WebFactCheckWorker:
        if speaker_id not in self.workers:
            self.workers[speaker_id] = WebFactCheckWorker(
                speaker_id, 
                self.on_fact_check_result_web
            )
        return self.workers[speaker_id]

    def on_fact_check_result_web(self, speaker_id: str, claim: str, result: dict, segment_id: str):
        # Store for analysis
        if speaker_id not in self.speaker_fact_checks:
            self.speaker_fact_checks[speaker_id] = []
        self.speaker_fact_checks[speaker_id].append(result["verdict"])

        # First do the normal logging
        self.on_fact_check_result(speaker_id, claim, result)
        
        # Then broadcast to web
        asyncio.run_coroutine_threadsafe(
        self.manager.broadcast({
            "type": "fact_check_result",
            "id": segment_id,
            "verdict": result["verdict"],
            "explanation": result["explanation"],
            "timestamp": time.time()  # Add this
        }),
        self.loop
        )

    # Overriding to broadcast progress
    def process_response(self, response):
        if not self.recording_active:
            return
        
        if not response:
            return
        
        words = []
        if hasattr(response, 'words'):
            words = response.words or []
        
        if not words:
            return
        
        segments = []
        current_speaker = None
        current_text = []
        
        for word in words:
            speaker = getattr(word, 'speaker_id', None)
            word_text = getattr(word, 'text', '')
            if getattr(word, 'type', 'word') != 'word':
                continue
            
            if speaker != current_speaker and current_text:
                segments.append({
                    "speaker": current_speaker,
                    "text": ' '.join(current_text)
                })
                current_text = []
            
            current_speaker = speaker
            current_text.append(word_text)
        
        if current_text:
            segments.append({
                "speaker": current_speaker,
                "text": ' '.join(current_text)
            })
        
        for segment in segments:
            speaker_id = segment["speaker"]
            text = segment["text"].strip()
            
            if not text or len(text) < 10:
                continue
            
            # ADD THIS LINE - force everything to be a statement
            text = force_statement(text)
    
            self.segment_counter += 1
            segment_id = f"seg_{self.segment_counter}_{int(time.time())}"
            
            # Store for final analysis
            self.transcript_for_analysis.append((speaker_id, text))

            # Broadcast transcript to UI
            asyncio.run_coroutine_threadsafe(
                self.manager.broadcast({
                    "type": "transcript",
                    "speaker_id": speaker_id,
                    "text": text,
                    "id": segment_id
                }),
                self.loop
            )

            # Local logging
            speaker_label, color = self.get_speaker_label(speaker_id)
            with self.print_lock:
                print(f"\n{color}{Colors.BOLD}{speaker_label}:{Colors.RESET} {text}")

            # Classify
            should_research = classify_statement(text)
            
            if should_research:
                # Notify UI research is starting
                asyncio.run_coroutine_threadsafe(
                    self.manager.broadcast({
                        "type": "fact_check_start",
                        "id": segment_id
                    }),
                    self.loop
                )
                
                # Add to worker
                worker = self.get_or_create_worker(speaker_id)
                worker.add_claim(text, segment_id)
            else:
                with self.print_lock:
                    print(f"  {Colors.DIM}[IGNORE - not a factual claim]{Colors.RESET}")

    async def generate_final_report(self):
        if not self.transcript_for_analysis:
            return {
                "type": "final_analysis",
                "summary": "No conversation recorded.",
                "speaker_1": {"trustworthiness": 100, "analysis": "N/A"},
                "speaker_2": {"trustworthiness": 100, "analysis": "N/A"}
            }
        
        # Add check for speaker_map
        if not self.speaker_map:
            return {
                "type": "final_analysis",
                "summary": "Conversation recorded but speakers could not be identified.",
                "speaker_1": {"trustworthiness": 100, "analysis": "Unable to identify speakers"},
                "speaker_2": {"trustworthiness": 100, "analysis": "N/A"}
            }

        # Format transcript
        full_transcript = ""
        for s_id, text in self.transcript_for_analysis:
            label, _ = self.get_speaker_label(s_id)
            full_transcript += f"{label}: {text}\n"

        # Calculate basic trustworthiness scores
        trust_scores = {}
        for s_id in self.speaker_map.keys():
            verdicts = self.speaker_fact_checks.get(s_id, [])
            if not verdicts:
                trust_scores[s_id] = 100.0
            else:
                truths = verdicts.count("TRUE")
                lies = verdicts.count("FALSE")
                partials = verdicts.count("PARTIAL")
                total = truths + lies + partials
                if total == 0:
                    trust_scores[s_id] = 100.0
                else:
                    # TRUE = 1, PARTIAL = 0.5, FALSE = 0
                    score = (truths * 1.0 + partials * 0.5) / total * 100
                    trust_scores[s_id] = score

        try:
            # Generate summary and analysis
            prompt = f"""Analyze the following conversation transcript. 
1. Provide a concise 2-3 sentence summary of the entire conversation.
2. Provide a brief (1-2 sentence) character analysis for each speaker based on their tone, claims, and style.

Transcript:
{full_transcript}

Respond in the following JSON format:
{{
  "summary": "...",
  "speaker_analysis": {{
    "Speaker 1": "...",
    "Speaker 2": "..."
  }}
}}
"""
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            
            # Map back to speaker IDs
            final_report = {
                "type": "final_analysis",
                "summary": data.get("summary", ""),
                "speaker_1": {
                    "trustworthiness": trust_scores.get(list(self.speaker_map.keys())[0] if self.speaker_map else None, 100),
                    "analysis": data.get("speaker_analysis", {}).get("Speaker 1", "No analysis available.")
                },
                "speaker_2": {
                    "trustworthiness": trust_scores.get(list(self.speaker_map.keys())[1] if len(self.speaker_map) > 1 else None, 100),
                    "analysis": data.get("speaker_analysis", {}).get("Speaker 2", "No analysis available.")
                }
            }
            return final_report

        except Exception as e:
            print(f"Error generating final report: {e}")
            return {
                "type": "final_analysis",
                "summary": "Error generating summary.",
                "speaker_1": {"trustworthiness": 0, "analysis": "Error"},
                "speaker_2": {"trustworthiness": 0, "analysis": "Error"}
            }

def run_transcriber(main_loop):
    # This runs in a background thread
    global transcriber
    transcriber = WebTranscriber(manager, main_loop)
    
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32',
            callback=transcriber.audio_callback,
        ):
            print(f"{Colors.GREEN}🎤 Web Transcriber Ready...{Colors.RESET}")
            
            last_transcription = time.time()
            while transcriber.running:
                time.sleep(0.1)
                # Only transcribe if recording is active
                if transcriber.recording_active and time.time() - last_transcription >= TRANSCRIPTION_INTERVAL:
                    audio_wav = transcriber.get_audio_as_wav()
                    if audio_wav and len(audio_wav) > 1000:
                        response = transcriber.transcribe_audio(audio_wav)
                        if response:
                            transcriber.process_response(response)
                    last_transcription = time.time()
                elif not transcriber.recording_active:
                    # Clear buffer if not recording
                    transcriber.get_audio_as_wav()
                    last_transcription = time.time()
                    
    except Exception as e:
        print(f"Transcriber error: {e}")
    finally:
        if transcriber:
            for worker in transcriber.workers.values():
                worker.stop()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start transcriber in a background thread
    loop = asyncio.get_running_loop()
    t = threading.Thread(target=run_transcriber, args=(loop,), daemon=True)
    t.start()
    yield
    # Shutdown: (Cleanup if needed)
    pass

app = FastAPI(lifespan=lifespan)

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Wait for messages from the frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "start":
                print("Recording started via UI")
                if transcriber:
                    transcriber.recording_active = True
                    # Clear previous session data if any
                    transcriber.transcript_for_analysis = []
                    transcriber.speaker_fact_checks = {}
            
            elif message.get("type") == "end":
                print("Recording ended via UI")
                if transcriber:
                    transcriber.recording_active = False
                    # Generate and broadcast final report
                    report = await transcriber.generate_final_report()
                    await manager.broadcast(report)
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Serve static files
@app.get("/")
async def get():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

# Mount static directory for other assets if any
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    # Start FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)

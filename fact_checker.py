#!/usr/bin/env python3
"""
Real-time Fact Checker with Speaker Diarization
================================================

Combines:
- ElevenLabs Scribe v2 for transcription with speaker detection
- OpenAI for classifying statements (RESEARCH vs IGNORE)
- Perplexity for fact-checking verified claims

Usage:
1. Set environment variables in .env:
   - ELEVENLABS_API_KEY
   - OPENAI_API_KEY
   - PPLX_API_KEY

2. Run: python fact_checker.py
"""

import os
import sys
import io
import wave
import time
import signal
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

import numpy as np
import sounddevice as sd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1")

from elevenlabs import ElevenLabs
from openai import OpenAI

# Import our modules
from classifier import classify_statement
from search import search

# OpenAI client for title generation
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# Configuration
# ============================================================

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PPLX_API_KEY = os.getenv("PPLX_API_KEY")

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1

# How often to transcribe audio (seconds) - still needed for API calls
TRANSCRIPTION_INTERVAL = 3.0

# Sentence batching - process after this many complete sentences per speaker
SENTENCES_PER_BATCH = 2

# Sentence ending punctuation
SENTENCE_ENDINGS = {'.', '!', '?'}

# Generate conversation title after this many lines
LINES_FOR_TITLE = 10

# Terminal colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    WHITE = '\033[97m'

SPEAKER_COLORS = [Colors.CYAN, Colors.GREEN, Colors.YELLOW, Colors.BLUE, Colors.HEADER]


class FactCheckWorker:
    """Worker that processes claims for a specific speaker."""
    
    def __init__(self, speaker_id: str, result_callback):
        self.speaker_id = speaker_id
        self.queue = Queue()
        self.result_callback = result_callback
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def add_claim(self, text: str):
        """Add a claim to be fact-checked."""
        self.queue.put(text)
    
    def _run(self):
        """Worker loop - processes claims from queue."""
        while self.running:
            try:
                claim = self.queue.get(timeout=0.5)
                result = self._fact_check(claim)
                self.result_callback(self.speaker_id, claim, result)
            except Empty:
                continue
            except Exception as e:
                print(f"{Colors.RED}[Worker Error] {e}{Colors.RESET}", file=sys.stderr)
    
    def _fact_check(self, claim: str) -> dict:
        """Fact-check a claim using Perplexity."""
        try:
            prompt = f"""Is the following statement true, false, or partially true? 
Provide a brief verdict (TRUE, FALSE, or PARTIAL) on the first line, then a one-sentence explanation.

Statement: "{claim}"
"""
            response = search(prompt, model="sonar")
            answer = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Parse verdict
            answer_upper = answer.upper()
            if "TRUE" in answer_upper[:50] and "FALSE" not in answer_upper[:20]:
                verdict = "TRUE"
            elif "FALSE" in answer_upper[:50]:
                verdict = "FALSE"
            elif "PARTIAL" in answer_upper[:50]:
                verdict = "PARTIAL"
            else:
                verdict = "UNCLEAR"
            
            return {
                "verdict": verdict,
                "explanation": answer,
                "success": True
            }
            
        except Exception as e:
            return {
                "verdict": "ERROR",
                "explanation": str(e),
                "success": False
            }
    
    def stop(self):
        self.running = False


class FactCheckTranscriber:
    """Main transcription + fact-checking system."""
    
    def __init__(self):
        # Verify API keys
        missing = []
        if not ELEVENLABS_API_KEY:
            missing.append("ELEVENLABS_API_KEY")
        if not OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not PPLX_API_KEY:
            missing.append("PPLX_API_KEY")
        
        if missing:
            print(f"{Colors.RED}Missing API keys: {', '.join(missing)}{Colors.RESET}")
            print(f"{Colors.YELLOW}Add them to your .env file{Colors.RESET}")
            sys.exit(1)
        
        self.client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        self.audio_buffer = []
        self.audio_lock = threading.Lock()
        self.running = True
        self.speaker_map = {}
        self.workers = {}  # speaker_id -> FactCheckWorker
        self.pending_checks = {}  # claim -> speaker for display
        
        # Sentence buffers per speaker - accumulate until we have complete sentences
        self.sentence_buffers = {}  # speaker_id -> {"text": str, "sentences": int}
        
        # Line counter and transcript history for title generation
        self.line_count = 0
        self.transcript_history = []  # List of {"speaker": str, "text": str}
        self.title_generated = False
        self.conversation_title = None
        
        # Output file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_file = f"factcheck_{timestamp}.txt"
        self.file_handle = open(self.output_file, "w", encoding="utf-8")
        self.file_handle.write(f"Fact-Check Session - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file_handle.write("=" * 70 + "\n\n")
        
        # Lock for thread-safe printing
        self.print_lock = threading.Lock()

    def get_speaker_color(self, speaker_num: int) -> str:
        return SPEAKER_COLORS[(speaker_num - 1) % len(SPEAKER_COLORS)]

    def get_speaker_label(self, speaker_id) -> tuple[str, str]:
        """Get speaker label and color."""
        if speaker_id is None:
            return "Unknown", Colors.WHITE
        if speaker_id not in self.speaker_map:
            self.speaker_map[speaker_id] = len(self.speaker_map) + 1
        num = self.speaker_map[speaker_id]
        return f"Speaker {num}", self.get_speaker_color(num)

    def get_or_create_worker(self, speaker_id) -> FactCheckWorker:
        """Get or create a fact-check worker for a speaker."""
        if speaker_id not in self.workers:
            self.workers[speaker_id] = FactCheckWorker(
                speaker_id, 
                self.on_fact_check_result
            )
        return self.workers[speaker_id]

    def count_sentences(self, text: str) -> int:
        """Count complete sentences in text."""
        count = 0
        for char in text:
            if char in SENTENCE_ENDINGS:
                count += 1
        return count

    def split_at_sentence(self, text: str, num_sentences: int) -> tuple[str, str]:
        """Split text after num_sentences complete sentences.
        Returns (complete_part, remainder)."""
        count = 0
        for i, char in enumerate(text):
            if char in SENTENCE_ENDINGS:
                count += 1
                if count == num_sentences:
                    return text[:i+1].strip(), text[i+1:].strip()
        return text, ""

    def get_sentence_buffer(self, speaker_id) -> dict:
        """Get or create sentence buffer for a speaker."""
        if speaker_id not in self.sentence_buffers:
            self.sentence_buffers[speaker_id] = {"text": "", "sentences": 0}
        return self.sentence_buffers[speaker_id]

    def generate_title(self):
        """Generate a title for the conversation based on transcript history."""
        if self.title_generated or not self.transcript_history:
            return
        
        # Build conversation text
        conversation_text = "\n".join([
            f"{entry['speaker']}: {entry['text']}" 
            for entry in self.transcript_history
        ])
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": f"""Based on this conversation, generate a short, descriptive title (5-8 words max). 
Return ONLY the title, nothing else.

Conversation:
{conversation_text}

Title:"""}
                ],
                max_tokens=30,
                temperature=0.7,
            )
            
            title = response.choices[0].message.content.strip().strip('"\'')
            self.conversation_title = title
            self.title_generated = True
            
            # Display the title
            with self.print_lock:
                print(f"\n{Colors.DIM}{'─' * 70}{Colors.RESET}")
                print(f"{Colors.BOLD}{Colors.HEADER}📌 CONVERSATION TITLE: {title}{Colors.RESET}")
                print(f"{Colors.DIM}{'─' * 70}{Colors.RESET}\n")
                
                # Write to file
                self.file_handle.write(f"\n{'─' * 70}\n")
                self.file_handle.write(f"📌 CONVERSATION TITLE: {title}\n")
                self.file_handle.write(f"{'─' * 70}\n\n")
                self.file_handle.flush()
                
        except Exception as e:
            print(f"{Colors.RED}[Title Generation Error] {e}{Colors.RESET}", file=sys.stderr)

    def process_complete_sentences(self, speaker_id: str, text: str):
        # Keep last sentence for context
        buffer = self.get_sentence_buffer(speaker_id)
        
        # Add previous sentence for context if available
        if hasattr(buffer, 'last_sentence') and buffer.last_sentence:
            contextualized_text = f"{buffer.last_sentence} {text}"
        else:
            contextualized_text = text
        """Process a batch of complete sentences for classification."""
        if not text or len(text) < 10:
            return
        
        speaker_label, color = self.get_speaker_label(speaker_id)
        
        # Track line count and history for title generation
        self.line_count += 1
        self.transcript_history.append({"speaker": speaker_label, "text": text})
        
        # Print the transcript segment
        with self.print_lock:
            print(f"\n{color}{Colors.BOLD}{speaker_label}:{Colors.RESET} {text}")
            self.file_handle.write(f"\n{speaker_label}: {text}\n")
            self.file_handle.flush()
        
        # Generate title after LINES_FOR_TITLE lines
        if self.line_count == LINES_FOR_TITLE and not self.title_generated:
            self.generate_title()
        
        # Classify the statement
        should_research = classify_statement(text)
        
        if should_research:
            with self.print_lock:
                print(f"  {Colors.BLUE}[🔍 RESEARCHING...]{Colors.RESET}")
            
            # Add to speaker's fact-check queue
            worker = self.get_or_create_worker(speaker_id)
            worker.add_claim(text)
        else:
            with self.print_lock:
                print(f"  {Colors.DIM}[IGNORE - not a factual claim]{Colors.RESET}")
        sentences = text.split('.')
        if sentences:
            buffer.last_sentence = sentences[-1].strip() + '.'

    def on_fact_check_result(self, speaker_id: str, claim: str, result: dict):
        """Callback when a fact-check completes."""
        with self.print_lock:
            speaker_label, color = self.get_speaker_label(speaker_id)
            verdict = result["verdict"]
            explanation = result["explanation"]
            
            # Verdict symbols and colors
            if verdict == "TRUE":
                symbol = "✓"
                verdict_color = Colors.GREEN
            elif verdict == "FALSE":
                symbol = "✗"
                verdict_color = Colors.RED
            elif verdict == "PARTIAL":
                symbol = "⚠"
                verdict_color = Colors.YELLOW
            else:
                symbol = "?"
                verdict_color = Colors.DIM
            
            # Print result
            print(f"\n{verdict_color}{symbol} {verdict}{Colors.RESET}: {claim[:60]}...")
            
            # Print brief explanation (first 150 chars)
            brief = explanation[:150].replace('\n', ' ')
            print(f"  {Colors.DIM}{brief}...{Colors.RESET}")
            
            # Write to file
            self.file_handle.write(f"\n[{verdict}] {claim}\n")
            self.file_handle.write(f"  {explanation[:300]}...\n")
            self.file_handle.flush()

    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback."""
        if status:
            print(f"{Colors.DIM}[Audio: {status}]{Colors.RESET}", file=sys.stderr)
        with self.audio_lock:
            self.audio_buffer.append(indata.copy())

    def get_audio_as_wav(self) -> bytes:
        """Get buffered audio as WAV bytes."""
        with self.audio_lock:
            if not self.audio_buffer:
                return None
            audio_data = np.concatenate(self.audio_buffer, axis=0)
            self.audio_buffer = []
        
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()

    def transcribe_audio(self, audio_wav: bytes):
        """Send audio to ElevenLabs for transcription."""
        try:
            audio_file = io.BytesIO(audio_wav)
            audio_file.name = "audio.wav"
            
            response = self.client.speech_to_text.convert(
                model_id="scribe_v2",
                file=audio_file,
                language_code="en",
                diarize=True,
                num_speakers=10,
                timestamps_granularity="word",
            )
            return response
        except Exception as e:
            print(f"{Colors.RED}Transcription error: {e}{Colors.RESET}", file=sys.stderr)
            return None

    def process_response(self, response):
        """Process transcription and accumulate sentences before classifying."""
        if not response:
            return
        
        words = []
        if hasattr(response, 'words'):
            words = response.words or []
        
        if not words:
            return
        
        # Group words by speaker into segments
        segments = []
        current_speaker = None
        current_text = []
        
        for word in words:
            speaker = getattr(word, 'speaker_id', None)
            word_text = getattr(word, 'text', '')
            word_type = getattr(word, 'type', 'word')
            
            if word_type != 'word':
                continue
            
            if speaker != current_speaker and current_text:
                segments.append({
                    "speaker": current_speaker,
                    "text": ' '.join(current_text)
                })
                current_text = []
            
            current_speaker = speaker
            current_text.append(word_text)
        
        # Don't forget the last segment
        if current_text:
            segments.append({
                "speaker": current_speaker,
                "text": ' '.join(current_text)
            })
        
        # Accumulate text per speaker and process when we have enough sentences
        for segment in segments:
            speaker_id = segment["speaker"]
            text = segment["text"].strip()
            
            if not text:
                continue
            
            # Get or create buffer for this speaker
            buffer = self.get_sentence_buffer(speaker_id)
            
            # Add new text to buffer
            if buffer["text"]:
                buffer["text"] += " " + text
            else:
                buffer["text"] = text
            
            # Count sentences in buffer
            sentence_count = self.count_sentences(buffer["text"])
            
            # Process if we have enough complete sentences
            while sentence_count >= SENTENCES_PER_BATCH:
                # Extract the complete sentences
                complete, remainder = self.split_at_sentence(buffer["text"], SENTENCES_PER_BATCH)
                
                # Process the complete sentences
                self.process_complete_sentences(speaker_id, complete)
                
                # Keep the remainder in the buffer
                buffer["text"] = remainder
                sentence_count = self.count_sentences(buffer["text"])

    def flush_sentence_buffers(self):
        """Process any remaining partial sentences in buffers."""
        for speaker_id, buffer in self.sentence_buffers.items():
            if buffer["text"] and len(buffer["text"].strip()) > 10:
                self.process_complete_sentences(speaker_id, buffer["text"].strip())
                buffer["text"] = ""

    def run(self):
        """Main loop."""
        print(f"{Colors.DIM}Available audio devices:{Colors.RESET}")
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                marker = " ← default" if i == default_input else ""
                print(f"{Colors.DIM}  [{i}] {dev['name']}{marker}{Colors.RESET}")
        print()
        
        print(f"{Colors.GREEN}📝 Saving to: {self.output_file}{Colors.RESET}")
        
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='float32',
                callback=self.audio_callback,
            ):
                print(f"{Colors.GREEN}🎤 Recording... (Ctrl+C to stop){Colors.RESET}")
                print(f"{Colors.DIM}Processing {SENTENCES_PER_BATCH} sentences at a time | Claims auto-verified{Colors.RESET}\n")
                print(f"{Colors.DIM}{'─' * 70}{Colors.RESET}")
                
                last_transcription = time.time()
                
                while self.running:
                    time.sleep(0.1)
                    
                    if time.time() - last_transcription >= TRANSCRIPTION_INTERVAL:
                        audio_wav = self.get_audio_as_wav()
                        
                        if audio_wav and len(audio_wav) > 1000:
                            response = self.transcribe_audio(audio_wav)
                            if response:
                                self.process_response(response)
                        
                        last_transcription = time.time()
                
        except sd.PortAudioError as e:
            print(f"{Colors.RED}Audio error: {e}{Colors.RESET}")
        finally:
            # Final transcription
            audio_wav = self.get_audio_as_wav()
            if audio_wav and len(audio_wav) > 1000:
                response = self.transcribe_audio(audio_wav)
                if response:
                    self.process_response(response)
            
            # Flush any remaining partial sentences
            self.flush_sentence_buffers()
            
            # Wait a moment for pending fact-checks to complete
            time.sleep(3)
            
            # Stop all workers
            for worker in self.workers.values():
                worker.stop()
            
            # Close file
            self.file_handle.write("\n" + "=" * 70 + "\n")
            self.file_handle.write(f"Session ended - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file_handle.close()
            
            print(f"\n\n{Colors.DIM}{'─' * 70}{Colors.RESET}")
            print(f"{Colors.GREEN}Session ended.{Colors.RESET}")
            if self.conversation_title:
                print(f"{Colors.HEADER}📌 Title: {self.conversation_title}{Colors.RESET}")
            print(f"{Colors.GREEN}📄 Results saved to: {self.output_file}{Colors.RESET}")

    def stop(self):
        self.running = False


def main():
    print(f"""
{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════╗
║            Real-time Fact Checker with Speaker Detection             ║
╠══════════════════════════════════════════════════════════════════════╣
║  🎤 ElevenLabs Scribe v2  │  🤖 OpenAI Classifier  │  🔍 Perplexity  ║
╚══════════════════════════════════════════════════════════════════════╝{Colors.RESET}
""")
    
    transcriber = FactCheckTranscriber()
    
    def signal_handler(sig, frame):
        print(f"\n{Colors.YELLOW}Stopping...{Colors.RESET}")
        transcriber.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        transcriber.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

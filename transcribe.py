#!/usr/bin/env python3
"""
ElevenLabs Scribe v2 Transcription with Speaker Diarization
============================================================

Captures audio from your MacBook microphone and transcribes it
using ElevenLabs Scribe v2 batch API with true speaker diarization.

Setup:
1. Create a .env file with: ELEVENLABS_API_KEY="your-api-key"
2. Install dependencies: pip install -r requirements.txt
3. Run: python transcribe.py
"""

import os
import sys
import io
import wave
import time
import signal
import threading
from queue import Queue

from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

# Get API key
API_KEY = os.getenv("ELEVENLABS_API_KEY")

if API_KEY:
    os.environ["ELEVENLABS_API_KEY"] = API_KEY

import numpy as np
import sounddevice as sd

# Suppress the pydantic warning
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1")

from elevenlabs import ElevenLabs

# ============================================================
# Configuration
# ============================================================

# Audio settings - 16kHz mono PCM
SAMPLE_RATE = 16000
CHANNELS = 1

# How often to send audio for transcription (in seconds)
TRANSCRIPTION_INTERVAL = 5.0  # Send every 5 seconds of audio

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

SPEAKER_COLORS = [Colors.CYAN, Colors.GREEN, Colors.YELLOW, Colors.BLUE, Colors.HEADER]


class BatchTranscriber:
    """Batch transcription using ElevenLabs Scribe v2 with speaker diarization."""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API_KEY is required")
        self.word_count = 0
        self.session_start = time.time()
        
        self.client = ElevenLabs(api_key=api_key)
        self.audio_buffer = []
        self.audio_lock = threading.Lock()
        self.running = True
        self.last_printed_speaker = None
        self.speaker_map = {}
        
        # Create output file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_file = f"transcript_{timestamp}.txt"
        self.file_handle = open(self.output_file, "w", encoding="utf-8")
        self.file_handle.write(f"Transcription Session - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.file_handle.write("=" * 60 + "\n\n")
        print(f"{Colors.GREEN}📝 Saving transcripts to: {self.output_file}{Colors.RESET}")
        self.last_audio_time = time.time()
        self.silence_threshold = 10.0  # 10 seconds of silence = new topic

    def get_speaker_color(self, speaker_num: int) -> str:
        return SPEAKER_COLORS[(speaker_num - 1) % len(SPEAKER_COLORS)]

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input."""
        if status:
            print(f"{Colors.DIM}[Audio: {status}]{Colors.RESET}", file=sys.stderr)
        
        # Detect if audio has actual content (not silence)
        audio_level = np.abs(indata).mean()
        
        with self.audio_lock:
            self.audio_buffer.append(indata.copy())
            
            # Update last audio time if we detect sound
            if audio_level > 0.01:  # Adjust threshold as needed
                current_time = time.time()
                silence_duration = current_time - self.last_audio_time
                
                # If there was a long pause, mark it
                if silence_duration > self.silence_threshold and self.last_printed_speaker:
                    print(f"\n{Colors.DIM}{'─' * 40} ({int(silence_duration)}s pause) {'─' * 40}{Colors.RESET}\n")
                    self.file_handle.write(f"\n{'─' * 40} ({int(silence_duration)}s pause) {'─' * 40}\n\n")
                    self.file_handle.flush()
                self.last_audio_time = current_time

    def get_audio_as_wav(self) -> bytes:
        """Get buffered audio as WAV bytes and clear buffer."""
        with self.audio_lock:
            if not self.audio_buffer:
                return None
            
            # Concatenate all audio chunks
            audio_data = np.concatenate(self.audio_buffer, axis=0)
            self.audio_buffer = []  # ✓ Already clearing - good!
        
        # Add length check to prevent memory issues
        max_length = SAMPLE_RATE * 60  # 60 seconds max
        if len(audio_data) > max_length:
            print(f"{Colors.YELLOW}[Warning: Audio buffer exceeded 60s, truncating]{Colors.RESET}")
            audio_data = audio_data[-max_length:]
        
        # Convert to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Create WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()

    def transcribe_audio(self, audio_wav: bytes, max_retries: int = 3):
        """Send audio to ElevenLabs Scribe v2 API for transcription."""
        for attempt in range(max_retries):
            try:
                audio_file = io.BytesIO(audio_wav)
                audio_file.name = "audio.wav"
                
                response = self.client.speech_to_text.convert(
                    model_id="scribe_v2",
                    file=audio_file,
                    language_code="en",
                    diarize=True,
                    num_speakers=2,
                    timestamps_granularity="word",
                )
                
                return response
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    print(f"{Colors.YELLOW}Transcription failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...{Colors.RESET}")
                    time.sleep(wait_time)
                else:
                    print(f"{Colors.RED}Transcription error after {max_retries} attempts: {e}{Colors.RESET}", file=sys.stderr)
                    return None

    def process_response(self, response):
        """Process transcription response and print with speaker labels."""
        if not response:
            return
        
        # Debug: print full response structure
        print(f"{Colors.DIM}[Response type: {type(response)}]{Colors.RESET}", file=sys.stderr)
        
        # Handle different response formats
        words = []
        text = ""
        
        if hasattr(response, 'words'):
            words = response.words or []
        elif hasattr(response, 'transcript'):
            text = response.transcript or ""
        elif isinstance(response, dict):
            words = response.get('words', [])
            text = response.get('transcript', response.get('text', ''))
        
        # If we have word-level data with speakers
        if words:
            current_speaker = None
            current_text = []
            
            # Debug: check first word structure
            if words:
                first_word = words[0]
                print(f"{Colors.DIM}[First word type: {type(first_word)}, attrs: {dir(first_word) if hasattr(first_word, '__dir__') else 'N/A'}]{Colors.RESET}", file=sys.stderr)
                if hasattr(first_word, 'speaker_id'):
                    print(f"{Colors.GREEN}[Speaker ID found: {first_word.speaker_id}]{Colors.RESET}", file=sys.stderr)
            
            for word in words:
                # Get speaker from word - field is 'speaker_id' not 'speaker'
                speaker = None
                if hasattr(word, 'speaker_id'):
                    speaker = word.speaker_id
                elif hasattr(word, 'speaker'):
                    speaker = word.speaker
                elif isinstance(word, dict):
                    speaker = word.get('speaker_id') or word.get('speaker')
                
                # Get word text
                if hasattr(word, 'text'):
                    word_text = word.text
                elif isinstance(word, dict):
                    word_text = word.get('text', '')
                else:
                    word_text = str(word)
                
                # Skip non-word types (spacing, audio_events)
                word_type = getattr(word, 'type', None) or (word.get('type') if isinstance(word, dict) else None)
                if word_type and word_type != 'word':
                    continue
                
                # Check for speaker change
                if speaker != current_speaker and current_text:
                    self.print_segment(current_speaker, ' '.join(current_text))
                    current_text = []
                
                current_speaker = speaker
                current_text.append(word_text)
            
            # Print remaining text
            if current_text:
                self.print_segment(current_speaker, ' '.join(current_text))
        
        elif text:
            # No word-level data, just print the full text
            self.print_segment(None, text)

    def print_segment(self, speaker_id, text: str):
        """Print a text segment with speaker label."""
        text = text.strip()
        if not text:
            return
        
        # Map speaker ID to number
        if speaker_id is not None:
            if speaker_id not in self.speaker_map:
                self.speaker_map[speaker_id] = len(self.speaker_map) + 1
            speaker_num = self.speaker_map[speaker_id]
            speaker_label = f"Speaker {speaker_num}"
            color = self.get_speaker_color(speaker_num)
        else:
            speaker_label = "Unknown"
            color = Colors.RESET
        
        # Always print speaker label on new line when speaker changes
        if speaker_label != self.last_printed_speaker:
            # Add newline before new speaker (except for first speaker)
            if self.last_printed_speaker is not None:
                print()
                self.file_handle.write("\n")
            
            print(f"{color}{Colors.BOLD}{speaker_label}:{Colors.RESET} {text}")
            self.file_handle.write(f"{speaker_label}: {text}\n")
            self.last_printed_speaker = speaker_label
        else:
            # Same speaker continuing - just append
            print(f"{text}", end=" ", flush=True)
            self.file_handle.write(f"{text} ")
        
        self.file_handle.flush()
        #Count words
        self.word_count += len(text.split())

    def run(self):
        """Main loop - capture audio and periodically transcribe."""
        # Show available devices
        print(f"{Colors.DIM}Available audio devices:{Colors.RESET}")
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                marker = " ← default" if i == default_input else ""
                print(f"{Colors.DIM}  [{i}] {dev['name']}{marker}{Colors.RESET}")
        print()
        
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='float32',
                callback=self.audio_callback,
            ):
                print(f"{Colors.GREEN}🎤 Recording... (Ctrl+C to stop){Colors.RESET}")
                print(f"{Colors.DIM}Transcription will appear every {TRANSCRIPTION_INTERVAL} seconds{Colors.RESET}\n")
                print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}\n")
                
                last_transcription = time.time()
                session_start = time.time()
                
                while self.running:
                    time.sleep(0.1)
                    
                    if time.time() - last_transcription >= TRANSCRIPTION_INTERVAL:
                        audio_wav = self.get_audio_as_wav()
                        if audio_wav and len(audio_wav) > 1000:
                            # Add timestamp every 5 minutes
                            elapsed = time.time() - session_start
                            if int(elapsed) % 300 == 0 and int(elapsed) > 0:  # Every 5 mins
                                timestamp = time.strftime("%H:%M:%S")
                                print(f"\n{Colors.DIM}[{timestamp} - {int(elapsed/60)} minutes elapsed]{Colors.RESET}\n")
                                self.file_handle.write(f"\n[{timestamp} - {int(elapsed/60)} minutes elapsed]\n\n")
                            
                            print(f"{Colors.DIM}[Transcribing...]{Colors.RESET}", end="", flush=True)
                            response = self.transcribe_audio(audio_wav)
                            print(f"\r{' ' * 20}\r", end="")
                            
                            if response:
                                self.process_response(response)
                        
                        last_transcription = time.time()
                
        except sd.PortAudioError as e:
            print(f"{Colors.RED}Audio device error: {e}{Colors.RESET}")
        finally:
            # Final transcription of remaining audio
            audio_wav = self.get_audio_as_wav()
            if audio_wav and len(audio_wav) > 1000:
                print(f"\n{Colors.DIM}[Final transcription...]{Colors.RESET}")
                response = self.transcribe_audio(audio_wav)
                if response:
                    self.process_response(response)
            
            # Calculate statistics
            duration = time.time() - self.session_start
            num_speakers = len(self.speaker_map)
            
            # Close file with statistics
            self.file_handle.write("\n\n" + "=" * 60 + "\n")
            self.file_handle.write(f"Session ended - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file_handle.write(f"Duration: {int(duration/60)} minutes {int(duration%60)} seconds\n")
            self.file_handle.write(f"Speakers detected: {num_speakers}\n")
            self.file_handle.write(f"Total words: {self.word_count}\n")
            self.file_handle.close()
            
            print(f"\n\n{Colors.DIM}{'─' * 60}{Colors.RESET}")
            print(f"{Colors.GREEN}Transcription ended.{Colors.RESET}")
            print(f"{Colors.CYAN}Duration: {int(duration/60)}m {int(duration%60)}s | Speakers: {num_speakers} | Words: {self.word_count}{Colors.RESET}")
            print(f"{Colors.GREEN}📄 Transcript saved to: {self.output_file}{Colors.RESET}")
    def stop(self):
        self.running = False


def main():
    if not API_KEY:
        print(f"""
{Colors.RED}╔══════════════════════════════════════════════════════════╗
║  Missing ELEVENLABS_API_KEY!                              ║
╠══════════════════════════════════════════════════════════╣
║  Create a .env file with:                                ║
║  ELEVENLABS_API_KEY="your-api-key-here"                  ║
║                                                          ║
║  Get your key at: https://elevenlabs.io/app/settings/api ║
╚══════════════════════════════════════════════════════════╝{Colors.RESET}
""")
        sys.exit(1)
    
    print(f"""
{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════╗
║   ElevenLabs Scribe v2 - Speaker Diarization             ║
╚══════════════════════════════════════════════════════════╝{Colors.RESET}
""")
    
    transcriber = BatchTranscriber(api_key=API_KEY)
    
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

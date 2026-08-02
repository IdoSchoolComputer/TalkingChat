import io
import wave
import queue
import os
import numpy as np
import sounddevice as sd
from tqdm import tqdm
from pathlib import Path
import asyncio

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    # torch not installed - faster-whisper can still run on CPU without it,
    # we just can't check for a GPU, so assume there isn't one.
    torch = None
    _HAS_TORCH = False

from groq import Groq
from faster_whisper import WhisperModel

try:
    from bidi.algorithm import get_display
    _HAS_BIDI = True
except ImportError:
    # python-bidi isn't installed - just print the raw string. It'll look
    # reversed/garbled in some terminals, but the script still runs.
    get_display = None
    _HAS_BIDI = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")          # required - used for language detection + English transcription
if not GROQ_API_KEY:
    raise ValueError("API key not found in environment variables")
GROQ_MODEL = "whisper-large-v3-turbo"   # fast Groq model, used for detection + non-Hebrew transcription
LOCAL_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"  # ivrit-ai's fastest Hebrew fine-tune (turbo)
SAMPLE_RATE = 16000

# Languages that need right-to-left display fixing in the terminal.
# (Whisper/Groq return ISO 639-1 codes like "he", "ar" for these.)
RTL_LANGUAGES = {"he", "ar", "fa", "ur"}

# --- Silence detection ------------------------------------------------------
# RMS is measured on the normalized [-1, 1] float signal, so this threshold
# is independent of sample width. 0.01 is a fairly conservative cutoff -
# quiet speech is usually well above this; room tone/mic hiss is below it.
SILENCE_RMS_THRESHOLD = 0.01
# If less than this fraction of the recording is above the RMS threshold,
# treat the whole clip as silence even if a stray click/pop pushed the
# overall RMS up momentarily.
MIN_VOICED_FRACTION = 0.02

# Auto-stop recording after this many seconds of continuous silence,
# but only once the user has actually started speaking - otherwise it'd
# stop before anyone said anything.
SILENCE_STOP_DURATION = 2.0


def fix_rtl(text):
    """
    Fix RTL Hebrew text for terminal display only.
    Don't apply this if you're writing to docx/PDF later - those already
    handle bidi text correctly on their own.
    """
    if _HAS_BIDI:
        # Reorders Hebrew/Arabic characters so they display right-to-left
        # in terminals that don't handle bidi text themselves.
        return get_display(text)
    return text


def is_silent(audio_int16, rms_threshold=SILENCE_RMS_THRESHOLD,
              min_voiced_fraction=MIN_VOICED_FRACTION, frame_ms=30):
    """
    Decide whether a recording is effectively silence.

    Rather than just checking the single loudest sample (which a single
    click/pop can trigger), this chops the audio into short frames,
    computes the RMS energy of each frame, and checks what fraction of
    frames are actually "voiced" (above rms_threshold). If almost none
    of the recording has energy above the threshold, we call it silence.

    Returns True if the audio should be treated as silence.
    """
    if audio_int16 is None or len(audio_int16) == 0:
        return True

    audio_float = audio_int16.astype(np.float32) / 32768.0

    frame_len = max(1, int(SAMPLE_RATE * frame_ms / 1000))
    n_frames = len(audio_float) // frame_len
    if n_frames == 0:
        # Recording shorter than one frame - just check overall RMS.
        rms = np.sqrt(np.mean(audio_float ** 2))
        return rms < rms_threshold

    trimmed = audio_float[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(trimmed ** 2, axis=1))

    voiced_fraction = np.mean(frame_rms > rms_threshold)
    return voiced_fraction < min_voiced_fraction


def record_audio():
    """
    Record from mic until either:
      - the user has spoken and then gone quiet for SILENCE_STOP_DURATION
        seconds (auto-stop), or
      - the user hits Ctrl+C (manual stop, still works at any point).
    Returns int16 mono audio at 16kHz.
    """
    # sounddevice records on a background thread and hands us chunks via
    # this callback - we can't just "return" from it, so we push each
    # chunk onto a thread-safe queue and drain that queue on the main thread.
    audio_queue = queue.Queue()

    def callback(indata, frames, time, status):
        audio_queue.put(indata.copy())  # .copy() - sounddevice reuses this buffer internally

    print(f"Recording... speak now. Stops automatically after "
          f"{SILENCE_STOP_DURATION:.1f}s of silence (or press Ctrl+C).")
    buffer = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', callback=callback)
    stream.start()

    # We don't start counting silence until the user has actually said
    # something - otherwise it'd auto-stop immediately, before they even
    # got a word out, since the mic starts out silent by definition.
    has_spoken = False
    silence_duration = 0.0

    try:
        while True:
            chunk = audio_queue.get()
            buffer.append(chunk)

            # Per-chunk RMS on the normalized signal, same measure is_silent()
            # uses on the full clip - keeps the two silence checks consistent.
            chunk_float = chunk.flatten().astype(np.float32) / 32768.0
            chunk_rms = np.sqrt(np.mean(chunk_float ** 2))
            chunk_duration = len(chunk_float) / SAMPLE_RATE

            if chunk_rms > SILENCE_RMS_THRESHOLD:
                has_spoken = True
                silence_duration = 0.0  # any voiced chunk resets the silence clock
            elif has_spoken:
                silence_duration += chunk_duration
                if silence_duration >= SILENCE_STOP_DURATION:
                    print(f"\n{SILENCE_STOP_DURATION:.1f}s of silence detected - "
                          "stopping recording.")
                    break
            # else: still silent and nobody has spoken yet - keep waiting,
            # don't count this toward the silence timer.
    except KeyboardInterrupt:
        print("\nStopped recording.")
    finally:
        # Always stop/close the stream, even if something above threw.
        stream.stop()
        stream.close()

    if not buffer:
        return None  # user hit Ctrl+C before any audio came in

    # Each item in `buffer` is a small (N, 1) chunk - stack them into one
    # long array, then flatten from (samples, 1) to a flat (samples,) array.
    audio_int16 = np.concatenate(buffer, axis=0).flatten()
    duration = len(audio_int16) / SAMPLE_RATE
    print(f"Captured {duration:.1f}s of audio.")

    if is_silent(audio_int16):
        # Frame-based RMS check - catches "wrong/muted mic" and "recorded
        # dead air" cases that a single peak-amplitude check can miss.
        print("Warning: recording appears to be silence - check your mic input.")

    return audio_int16


def audio_to_wav_bytes(audio_int16):
    """Pack int16 mono audio into an in-memory WAV file for upload to Groq."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)       # mono
        wf.setsampwidth(2)       # 2 bytes = 16-bit samples, matches our int16 audio
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    buf.seek(0)              # rewind so the Groq client reads from the start
    buf.name = "audio.wav"   # the Groq SDK/requests lib uses this for the multipart filename
    return buf


def detect_language_via_groq(audio_int16):
    """
    Send audio to Groq once. Groq tells us what language it is, and
    hands back a transcript in the same call.

    - If Groq says Hebrew: we throw its transcript away (ivrit-ai's
      local model is more accurate for Hebrew) and let ivrit handle it.
    - If Groq says anything else (English etc.): its transcript is
      good enough, so we just use it directly - no local model needed.

    Returns (is_hebrew: bool, groq_text: str, detected_lang: str)
    """
    if not GROQ_API_KEY or GROQ_API_KEY == "not telling u":
        raise ValueError("Set GROQ_API_KEY at the top of the script.")

    client = Groq(api_key=GROQ_API_KEY)

    print("Detecting language via Groq...")
    transcript = client.audio.transcriptions.create(
        file=audio_to_wav_bytes(audio_int16),
        model=GROQ_MODEL,
        # No `language=` param here on purpose - leaving it unset makes
        # Whisper auto-detect instead of forcing one language.
        response_format="verbose_json",   # this is what makes the `.language` field come back
    )

    # transcript.language comes back as a full word like "english" / "hebrew",
    # not an ISO code - startswith("he") catches "hebrew" (and "he" just in case).
    detected_lang = (getattr(transcript, "language", "") or "").lower()
    is_hebrew = detected_lang.startswith("he")

    print(f"Groq detected language: {detected_lang or 'unknown'}")

    return is_hebrew, transcript.text, detected_lang


def load_local_model():
    """
    Load the local ivrit-ai model. Uses CUDA + float16 if a GPU is
    available (much faster), otherwise falls back to CPU + int8.
    """
    print(f"Loading local model: {LOCAL_MODEL} ...")

    has_cuda = _HAS_TORCH and torch.cuda.is_available()

    if has_cuda:
        # float16 needs a GPU - much faster than int8 on CPU for a model this size.
        model = WhisperModel(LOCAL_MODEL, device="cuda", compute_type="float16")
        print("Local model loaded on GPU (float16).")
    else:
        # int8 quantization keeps CPU inference reasonably fast/light on RAM.
        model = WhisperModel(LOCAL_MODEL, device="cpu", compute_type="int8")
        print("Local model loaded on CPU (int8).")

    return model


def transcribe_local_hebrew(audio_int16,HebrewModel):
    """
    Transcribe Hebrew audio using the local ivrit-ai model.
    Language is already known (Groq told us it's Hebrew), so we skip
    local language detection entirely and go straight to transcription -
    this alone saves a full extra pass over the audio.
    """

    # faster-whisper expects float32 samples in [-1, 1], not raw int16.
    audio_float = audio_int16.astype(np.float32) / 32768.0
    duration = len(audio_int16) / SAMPLE_RATE

    segments, info = HebrewModel.transcribe(
        audio_float,
        language="he",            # skip auto-detect - we already know it's Hebrew
        beam_size=5,               # wider beam search = more accurate, a bit slower
        vad_filter=True,           # strips silence/noise before it reaches the model
        condition_on_previous_text=False,  # stops it from looping/repeating on itself
    )
    # `segments` is a lazy generator - nothing has actually run yet, the model
    # only processes audio as we iterate over it below.

    output = ""
    last_end = 0.0
    # Progress bar tracked against audio duration (in seconds), not segment
    # count, since we don't know how many segments there'll be up front.
    with tqdm(total=round(duration, 1), unit="s", desc="Transcribing (ivrit)") as pbar:
        for segment in segments:
            output = ""+segment.text
            # Advance the bar by however much audio this segment covered.
            pbar.update(round(segment.end - last_end, 1))
            last_end = segment.end
        if last_end < duration:
            # VAD may have skipped trailing silence, leaving the bar short -
            # top it off so it always finishes at 100%.
            pbar.update(round(duration - last_end, 1))

    if not output:
        # vad_filter=True can legitimately strip an entire clip if it was
        # all silence/noise that our earlier is_silent() check let through
        # (e.g. RMS was borderline). Say so instead of printing nothing.
        return "No speech detected in audio."

    # Printed after the progress bar finishes so lines don't interleave with it.
    print(fix_rtl(output))
    return output


def record_and_transcribe(speak,HebrewModel,quit):
    audio = record_audio()
    if audio is None:
        return "No audio captured. Hence no input captured"
        

    if is_silent(audio):
        # Bail out before spending an API call on a clip that's just silence.
        asyncio.run(speak("Recording is silent - nothing to transcribe. Exiting."))
        quit()
        

    is_hebrew, groq_text, detected_lang = detect_language_via_groq(audio)

    if is_hebrew:
        # Hebrew: re-transcribe locally with ivrit-ai for better accuracy.
        print("Hebrew detected - handing off to the local ivrit model...")
        return transcribe_local_hebrew(audio,HebrewModel)
    elif detected_lang.startswith('en'):
        # Anything else: Groq's own transcript is already good, so just use it -
        # no need to spin up the local model at all.
        print("English detected - using Groq's transcript directly.")
        return groq_text
    else:
        asyncio.run(speak('Not English or Hebrew, stopping program...'))
        quit()
        
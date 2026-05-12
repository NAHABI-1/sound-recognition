from pathlib import Path

import speech_recognition as sr


class SpeechToTextError(Exception):
    """Raised when speech cannot be converted into text."""


def transcribe_audio(wav_path: Path) -> dict:
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(str(wav_path)) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio_data = recognizer.record(source)
    except Exception as exc:
        raise SpeechToTextError("Could not read the WAV file for speech recognition.") from exc

    try:
        text = recognizer.recognize_google(audio_data)
        return {
            "text": text,
            "status": "success",
            "engine": "Google Speech Recognition",
            "message": "Speech was transcribed successfully.",
        }
    except sr.UnknownValueError as exc:
        raise SpeechToTextError(
            "No clear speech was detected. Try a shorter, clearer recording with less background noise."
        ) from exc
    except sr.RequestError:
        return _try_offline_fallback(recognizer, audio_data)


def _try_offline_fallback(recognizer: sr.Recognizer, audio_data: sr.AudioData) -> dict:
    try:
        text = recognizer.recognize_sphinx(audio_data)
        return {
            "text": text,
            "status": "success",
            "engine": "PocketSphinx offline recognizer",
            "message": "Google recognition was unavailable, so offline recognition was used.",
        }
    except sr.UnknownValueError as exc:
        raise SpeechToTextError(
            "Speech was detected but could not be understood. The audio may be too noisy."
        ) from exc
    except (sr.RequestError, AttributeError, ModuleNotFoundError) as exc:
        raise SpeechToTextError(
            "Speech recognition service is unavailable. Check your internet connection, "
            "or install PocketSphinx for offline fallback."
        ) from exc

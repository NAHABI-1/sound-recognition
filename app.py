import os
import tempfile
from pathlib import Path

from flask import Flask, render_template, request

from utils.audio_processor import AudioProcessingError, prepare_audio_upload
from utils.fourier_analysis import FourierAnalysisError, analyze_audio
from utils.speech_to_text import SpeechToTextError, transcribe_audio


BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = Path(os.environ.get("AUDIO_WORK_DIR", tempfile.gettempdir())) / "sound-recognition"
UPLOAD_DIR = WORK_DIR / "uploads"
GRAPHS_DIR = WORK_DIR / "graphs"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_audio():
    file = request.files.get("audio")

    try:
        original_path, wav_path = prepare_audio_upload(file, UPLOAD_DIR)
        analysis = analyze_audio(wav_path, GRAPHS_DIR)
        warning = None

        try:
            transcription = transcribe_audio(wav_path)
        except SpeechToTextError as exc:
            warning = str(exc)
            transcription = {
                "text": "",
                "status": "warning",
                "engine": "SpeechRecognition",
                "message": warning,
            }

        result = {
            "original_filename": original_path.name,
            "wav_filename": wav_path.name,
            "transcription": transcription,
            "analysis": analysis,
        }
        return render_template("index.html", result=result, warning=warning)

    except AudioProcessingError as exc:
        return render_template("index.html", error=str(exc)), 400
    except FourierAnalysisError as exc:
        return render_template("index.html", error=str(exc)), 422
    except Exception:
        app.logger.exception("Unexpected error while processing uploaded audio")
        return render_template(
            "index.html",
            error="Something went wrong while processing the audio. Please try a different file.",
        ), 500


@app.errorhandler(413)
def file_too_large(_error):
    return render_template("index.html", error="File is too large. Maximum size is 25 MB."), 413


if __name__ == "__main__":
    app.run(debug=True)

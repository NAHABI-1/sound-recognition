from pathlib import Path
from shutil import which
import subprocess
from uuid import uuid4

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "webm", "ogg"}
MAX_UPLOAD_MB = 25


class AudioProcessingError(Exception):
    """Raised when an uploaded audio file cannot be validated or converted."""


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_audio_filename(filename: str) -> str:
    clean_name = secure_filename(filename or "")
    if not clean_name:
        raise AudioProcessingError("Please choose a valid audio file.")

    stem = Path(clean_name).stem[:80] or "audio"
    suffix = Path(clean_name).suffix.lower()
    return f"{stem}_{uuid4().hex[:10]}{suffix}"


def save_upload(file: FileStorage, upload_dir: Path) -> Path:
    if file is None or not file.filename:
        raise AudioProcessingError("No audio file was uploaded.")

    if not allowed_file(file.filename):
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise AudioProcessingError(f"Invalid file type. Please upload one of: {allowed}.")

    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_audio_filename(file.filename)
    destination = upload_dir / filename
    file.save(destination)

    if destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise AudioProcessingError("The uploaded file is empty.")

    size_mb = destination.stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        destination.unlink(missing_ok=True)
        raise AudioProcessingError(f"File is too large. Maximum size is {MAX_UPLOAD_MB} MB.")

    return destination


def convert_to_wav(source_path: Path, upload_dir: Path) -> Path:
    """Convert supported audio to a clean mono 16 kHz WAV for recognition."""
    if source_path.suffix.lower() == ".wav":
        return source_path

    wav_path = upload_dir / f"{source_path.stem}_processed.wav"
    ffmpeg = _get_ffmpeg_exe()

    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(source_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-sample_fmt",
                "s16",
                "-vn",
                str(wav_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        raise AudioProcessingError(
            "Could not prepare the audio for processing. Install FFmpeg or install the Python "
            "fallback with: pip install imageio-ffmpeg. The audio file may also be corrupted."
        ) from exc

    if not wav_path.exists() or wav_path.stat().st_size == 0:
        raise AudioProcessingError("Audio preparation failed and produced an empty WAV file.")

    return wav_path


def _get_ffmpeg_exe() -> str:
    """Return system FFmpeg or the imageio-ffmpeg bundled binary."""
    system_ffmpeg = which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise AudioProcessingError(
            "FFmpeg is required to convert MP3, M4A, WEBM, or OGG files to WAV. "
            "Run: pip install imageio-ffmpeg, or install system FFmpeg."
        ) from exc


def prepare_audio_upload(file: FileStorage, upload_dir: Path) -> tuple[Path, Path]:
    original_path = save_upload(file, upload_dir)
    wav_path = convert_to_wav(original_path, upload_dir)
    return original_path, wav_path

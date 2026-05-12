import base64
from html import escape
from pathlib import Path
import wave

import numpy as np


MAX_ANALYSIS_SECONDS = 12


class FourierAnalysisError(Exception):
    """Raised when Fourier analysis cannot be completed."""


def analyze_audio(wav_path: Path, graphs_dir: Path) -> dict:
    graphs_dir.mkdir(parents=True, exist_ok=True)
    signal, sample_rate = _load_signal(wav_path)

    duration = float(len(signal) / sample_rate)
    dominant_frequency, fft_frequencies, fft_magnitudes = _fft(signal, sample_rate)
    spectrogram_data = _spectrogram(signal, sample_rate)

    return {
        "duration": round(duration, 2),
        "sample_rate": int(sample_rate),
        "dominant_frequency": round(float(dominant_frequency), 2),
        "waveform_graph": _svg_data_uri(_waveform_svg(signal, sample_rate)),
        "fft_graph": _svg_data_uri(_fft_svg(fft_frequencies, fft_magnitudes, dominant_frequency)),
        "spectrogram_graph": _svg_data_uri(_spectrogram_svg(*spectrogram_data)),
    }


def _load_signal(wav_path: Path) -> tuple[np.ndarray, int]:
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())
    except Exception as exc:
        raise FourierAnalysisError("Could not load audio for Fourier analysis.") from exc

    signal = _pcm_to_float(frames, sample_width)
    if channels > 1:
        signal = signal.reshape(-1, channels).mean(axis=1)

    signal = signal[np.isfinite(signal)]
    if signal.size == 0:
        raise FourierAnalysisError("The audio file contains no usable signal.")

    if not sample_rate or sample_rate <= 0:
        raise FourierAnalysisError("The audio file has an invalid sample rate.")

    peak = np.max(np.abs(signal))
    if peak > 1:
        signal = signal / peak

    max_samples = int(sample_rate * MAX_ANALYSIS_SECONDS)
    if len(signal) > max_samples:
        signal = signal[:max_samples]

    return signal.astype(np.float32), int(sample_rate)


def _pcm_to_float(frames: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        signal = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        return (signal - 128) / 128

    if sample_width == 2:
        signal = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        return signal / np.iinfo(np.int16).max

    if sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        sign = (raw[:, 2] & 0x80) != 0
        padded = np.zeros((raw.shape[0], 4), dtype=np.uint8)
        padded[:, :3] = raw
        padded[sign, 3] = 0xFF
        signal = padded.view("<i4").reshape(-1).astype(np.float32)
        return signal / float(2**23 - 1)

    if sample_width == 4:
        signal = np.frombuffer(frames, dtype="<i4").astype(np.float32)
        return signal / np.iinfo(np.int32).max

    raise FourierAnalysisError("Unsupported WAV sample width.")


def _fft(signal: np.ndarray, sample_rate: int) -> tuple[float, np.ndarray, np.ndarray]:
    centered_signal = signal - float(np.mean(signal))
    if len(centered_signal) < 2 or np.max(np.abs(centered_signal)) < 1e-6:
        return 0.0, np.array([0.0, 1.0]), np.array([0.0, 0.0])

    windowed_signal = centered_signal * np.hanning(len(centered_signal))
    fft_values = np.abs(np.fft.rfft(windowed_signal))
    frequencies = np.fft.rfftfreq(len(windowed_signal), d=1 / sample_rate)
    dominant_index = int(np.argmax(fft_values[1:]) + 1) if len(fft_values) > 1 else 0
    dominant_frequency = frequencies[dominant_index]

    max_display_frequency = min(sample_rate / 2, 8000)
    display_mask = frequencies <= max_display_frequency
    display_frequencies = frequencies[display_mask]
    magnitudes = fft_values[display_mask]
    magnitudes = magnitudes / max(float(np.max(magnitudes)), 1e-12)
    return float(dominant_frequency), display_frequencies, magnitudes


def _spectrogram(
    signal: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(signal) < 2 or np.max(np.abs(signal - float(np.mean(signal)))) < 1e-6:
        return np.array([0.0]), np.array([0.0]), np.zeros((1, 1), dtype=np.float32)

    nperseg = min(512, len(signal))
    noverlap = min(nperseg // 2, nperseg - 1)
    step = max(nperseg - noverlap, 1)
    padded_signal = np.pad(signal, (0, max(0, nperseg - len(signal))))
    frame_count = 1 + max((len(padded_signal) - nperseg) // step, 0)
    starts = np.arange(frame_count) * step
    frames = np.stack([padded_signal[start : start + nperseg] for start in starts])
    frames = frames - frames.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(nperseg), axis=1)) ** 2
    frequencies = np.fft.rfftfreq(nperseg, d=1 / sample_rate)
    frequency_mask = frequencies <= min(sample_rate / 2, 8000)
    spectrum_db = 10 * np.log10(spectrum[:, frequency_mask] + 1e-12).T
    times = (starts + nperseg / 2) / sample_rate
    return frequencies[frequency_mask], times, spectrum_db


def _waveform_svg(signal: np.ndarray, sample_rate: int) -> str:
    display_signal = _sample_evenly(signal, 900)
    times = np.linspace(0, len(signal) / sample_rate, len(display_signal))
    points = _line_points(times, display_signal, 80, 720, 70, 250, y_min=-1, y_max=1)
    return _svg_chart(
        "Time-Domain Waveform",
        "Time (seconds)",
        "Amplitude",
        f'<polyline points="{points}" fill="none" stroke="#0ea5e9" stroke-width="2" />',
    )


def _fft_svg(frequencies: np.ndarray, magnitudes: np.ndarray, dominant_frequency: float) -> str:
    display_frequencies = _sample_evenly(frequencies, 900)
    display_magnitudes = _sample_evenly(magnitudes, 900)
    points = _line_points(display_frequencies, display_magnitudes, 80, 720, 70, 250, y_min=0, y_max=1)
    dominant_x = _scale(dominant_frequency, 0, max(float(frequencies[-1]), 1), 80, 720)
    body = (
        f'<polyline points="{points}" fill="none" stroke="#16a34a" stroke-width="2" />'
        f'<line x1="{dominant_x:.2f}" x2="{dominant_x:.2f}" y1="70" y2="250" '
        'stroke="#dc2626" stroke-width="2" stroke-dasharray="6 6" />'
    )
    return _svg_chart("Frequency-Domain Fourier Spectrum", "Frequency (Hz)", "Magnitude", body)


def _spectrogram_svg(frequencies: np.ndarray, times: np.ndarray, spectrum_db: np.ndarray) -> str:
    rows = min(48, spectrum_db.shape[0])
    cols = min(90, spectrum_db.shape[1])
    reduced = spectrum_db
    if spectrum_db.shape[0] > rows:
        reduced = reduced[np.linspace(0, spectrum_db.shape[0] - 1, rows).astype(int), :]
    if spectrum_db.shape[1] > cols:
        reduced = reduced[:, np.linspace(0, spectrum_db.shape[1] - 1, cols).astype(int)]

    minimum = float(np.min(reduced))
    maximum = float(np.max(reduced))
    span = max(maximum - minimum, 1e-9)
    cell_width = 640 / max(reduced.shape[1], 1)
    cell_height = 180 / max(reduced.shape[0], 1)
    rects = []
    for row in range(reduced.shape[0]):
        for col in range(reduced.shape[1]):
            level = (float(reduced[row, col]) - minimum) / span
            rects.append(
                '<rect '
                f'x="{80 + col * cell_width:.2f}" '
                f'y="{70 + (reduced.shape[0] - row - 1) * cell_height:.2f}" '
                f'width="{cell_width + 0.5:.2f}" height="{cell_height + 0.5:.2f}" '
                f'fill="{_heat_color(level)}" />'
            )

    body = "".join(rects)
    return _svg_chart("Spectrogram", "Time (seconds)", "Frequency (Hz)", body)


def _sample_evenly(values: np.ndarray, max_points: int) -> np.ndarray:
    if len(values) <= max_points:
        return values
    indexes = np.linspace(0, len(values) - 1, max_points).astype(int)
    return values[indexes]


def _line_points(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x1: int,
    x2: int,
    y1: int,
    y2: int,
    *,
    y_min: float | None = None,
    y_max: float | None = None,
) -> str:
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    if y_min is None:
        y_min = float(np.min(y_values))
    if y_max is None:
        y_max = float(np.max(y_values))

    points = []
    for x_value, y_value in zip(x_values, y_values):
        x = _scale(float(x_value), x_min, x_max, x1, x2)
        y = _scale(float(y_value), y_min, y_max, y2, y1)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _scale(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
    if abs(source_max - source_min) < 1e-12:
        return (target_min + target_max) / 2
    ratio = (value - source_min) / (source_max - source_min)
    ratio = max(0, min(1, ratio))
    return target_min + ratio * (target_max - target_min)


def _heat_color(level: float) -> str:
    level = max(0, min(1, level))
    red = int(30 + level * 225)
    green = int(20 + max(0, level - 0.35) * 170)
    blue = int(90 + (1 - level) * 100)
    return f"rgb({red},{green},{blue})"


def _svg_chart(title: str, x_label: str, y_label: str, body: str) -> str:
    safe_title = escape(title)
    safe_x_label = escape(x_label)
    safe_y_label = escape(y_label)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 320" role="img" aria-label="{safe_title}">
  <rect width="800" height="320" fill="#ffffff"/>
  <text x="400" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#0f172a">{safe_title}</text>
  <line x1="80" y1="250" x2="720" y2="250" stroke="#94a3b8" stroke-width="1"/>
  <line x1="80" y1="70" x2="80" y2="250" stroke="#94a3b8" stroke-width="1"/>
  <g opacity="0.25" stroke="#64748b" stroke-width="1">
    <line x1="80" y1="205" x2="720" y2="205"/>
    <line x1="80" y1="160" x2="720" y2="160"/>
    <line x1="80" y1="115" x2="720" y2="115"/>
  </g>
  {body}
  <text x="400" y="296" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#334155">{safe_x_label}</text>
  <text x="25" y="160" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#334155" transform="rotate(-90 25 160)">{safe_y_label}</text>
</svg>"""


def _svg_data_uri(svg: str) -> str:
    svg_data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{svg_data}"

from pathlib import Path
from uuid import uuid4

import matplotlib
import numpy as np
import soundfile as sf

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import spectrogram


class FourierAnalysisError(Exception):
    """Raised when Fourier analysis cannot be completed."""


def analyze_audio(wav_path: Path, graphs_dir: Path) -> dict:
    graphs_dir.mkdir(parents=True, exist_ok=True)
    graph_id = uuid4().hex[:10]

    signal, sample_rate = _load_signal(wav_path)

    duration = float(len(signal) / sample_rate)
    waveform_path = graphs_dir / f"waveform_{graph_id}.png"
    fft_path = graphs_dir / f"fft_{graph_id}.png"
    spectrogram_path = graphs_dir / f"spectrogram_{graph_id}.png"

    _save_waveform(signal, sample_rate, waveform_path)
    dominant_frequency = _save_fft(signal, sample_rate, fft_path)
    _save_spectrogram(signal, sample_rate, spectrogram_path)

    return {
        "duration": round(duration, 2),
        "sample_rate": int(sample_rate),
        "dominant_frequency": round(float(dominant_frequency), 2),
        "waveform_graph": f"graphs/{waveform_path.name}",
        "fft_graph": f"graphs/{fft_path.name}",
        "spectrogram_graph": f"graphs/{spectrogram_path.name}",
    }


def _load_signal(wav_path: Path) -> tuple[np.ndarray, int]:
    try:
        signal, sample_rate = sf.read(str(wav_path), always_2d=False)
    except Exception as exc:
        raise FourierAnalysisError("Could not load audio for Fourier analysis.") from exc

    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)

    signal = signal[np.isfinite(signal)]
    if signal.size == 0:
        raise FourierAnalysisError("The audio file contains no usable signal.")

    if not sample_rate or sample_rate <= 0:
        raise FourierAnalysisError("The audio file has an invalid sample rate.")

    peak = np.max(np.abs(signal))
    if peak > 1:
        signal = signal / peak

    return signal.astype(np.float32), int(sample_rate)


def _save_waveform(signal: np.ndarray, sample_rate: int, output_path: Path) -> None:
    display_signal = _decimate_for_plot(signal)
    times = np.arange(len(display_signal)) * (len(signal) / len(display_signal)) / sample_rate

    plt.figure(figsize=(11, 4.5), dpi=140)
    plt.plot(times, display_signal, color="#38bdf8", linewidth=0.8)
    plt.title("Time-Domain Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.ylim(-1.05, 1.05)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, transparent=False, facecolor="white")
    plt.close()


def _save_fft(signal: np.ndarray, sample_rate: int, output_path: Path) -> float:
    centered_signal = signal - float(np.mean(signal))
    if len(centered_signal) < 2 or np.max(np.abs(centered_signal)) < 1e-6:
        _save_empty_frequency_plot(output_path, "Frequency-Domain Fourier Spectrum")
        return 0.0

    windowed_signal = centered_signal * np.hanning(len(centered_signal))
    fft_values = np.abs(rfft(windowed_signal))
    frequencies = rfftfreq(len(windowed_signal), d=1 / sample_rate)

    if len(fft_values) > 1:
        dominant_index = int(np.argmax(fft_values[1:]) + 1)
    else:
        dominant_index = 0
    dominant_frequency = frequencies[dominant_index]

    max_display_frequency = min(sample_rate / 2, 8000)
    display_mask = frequencies <= max_display_frequency
    magnitudes = fft_values / max(np.max(fft_values), 1e-12)

    plt.figure(figsize=(11, 4.5), dpi=140)
    plt.plot(frequencies[display_mask], magnitudes[display_mask], color="#22c55e", linewidth=0.9)
    plt.axvline(dominant_frequency, color="#ef4444", linestyle="--", linewidth=1.1, label="Dominant frequency")
    plt.title("Frequency-Domain Fourier Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized magnitude")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, transparent=False, facecolor="white")
    plt.close()

    return float(dominant_frequency)


def _save_spectrogram(signal: np.ndarray, sample_rate: int, output_path: Path) -> None:
    if len(signal) < 2 or np.max(np.abs(signal - float(np.mean(signal)))) < 1e-6:
        _save_empty_frequency_plot(output_path, "Spectrogram")
        return

    nperseg = min(1024, len(signal))
    noverlap = min(nperseg // 2, nperseg - 1)
    frequencies, times, spectrum = spectrogram(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
    )
    spectrum_db = 10 * np.log10(spectrum + 1e-12)

    plt.figure(figsize=(11, 4.8), dpi=140)
    plt.pcolormesh(times, frequencies, spectrum_db, shading="gouraud", cmap="magma")
    plt.title("Spectrogram")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, min(sample_rate / 2, 8000))
    colorbar = plt.colorbar()
    colorbar.set_label("Magnitude (dB)")
    plt.tight_layout()
    plt.savefig(output_path, transparent=False, facecolor="white")
    plt.close()


def _decimate_for_plot(signal: np.ndarray, max_points: int = 50_000) -> np.ndarray:
    if len(signal) <= max_points:
        return signal

    step = int(np.ceil(len(signal) / max_points))
    return signal[::step]


def _save_empty_frequency_plot(output_path: Path, title: str) -> None:
    plt.figure(figsize=(11, 4.5), dpi=140)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.text(0.5, 0.5, "No measurable frequency energy", ha="center", va="center", transform=plt.gca().transAxes)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, transparent=False, facecolor="white")
    plt.close()

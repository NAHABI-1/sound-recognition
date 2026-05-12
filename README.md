# PYTHON SOUND RECOGNITION SYSTEM WITH FOURIER ANALYSIS

A Flask web application that lets users upload or record audio, converts speech to text, and displays Fourier-based signal analysis graphs.

## 1. Project Overview

This project demonstrates how speech/audio can be processed in Python using signal processing and speech recognition techniques. Users can upload an audio file or record audio in the browser, then the system:

- Converts the audio into a WAV format when needed.
- Extracts duration and sampling rate.
- Transcribes speech into text using the `SpeechRecognition` library.
- Generates waveform, FFT spectrum, and spectrogram graphs.
- Displays the dominant frequency of the audio signal.

## 2. How the System Works

1. The user uploads or records an audio file from the web interface.
2. Flask receives the file through the `/upload` route.
3. `audio_processor.py` validates the file, saves it, and converts MP3, M4A, WEBM, or OGG files to WAV.
4. `speech_to_text.py` reads the WAV file and sends it to Google Speech Recognition. If Google is unavailable, it attempts an offline PocketSphinx fallback if installed.
5. `fourier_analysis.py` loads the audio with `librosa`, calculates signal metadata, applies FFT, detects the dominant frequency, and saves graph images.
6. The results are rendered back in `index.html`.

## 3. Role of Fourier Transform in Sound Recognition

Audio is a time-domain signal, meaning it changes over time. Speech recognition and sound analysis become easier when the signal is also studied in the frequency domain.

Fourier Transform breaks a complex sound wave into its frequency components. In this project:

- The waveform graph shows amplitude over time.
- The FFT graph shows which frequencies are present in the sound and how strong they are.
- The dominant frequency is calculated from the largest FFT magnitude peak.
- The spectrogram shows how frequency energy changes over time.

This helps identify pitch, speech energy, noise, and frequency patterns that are important in sound recognition.

## 4. Installation Steps

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

MP3, M4A, WEBM, and OGG conversion uses `pydub`, which requires FFmpeg on your computer.

On Ubuntu/Debian:

```bash
sudo apt install ffmpeg
```

On macOS with Homebrew:

```bash
brew install ffmpeg
```

## 5. How to Run the Project

From the project folder:

```bash
python app.py
```

Open the Flask URL shown in the terminal, usually:

```text
http://127.0.0.1:5000
```

## 6. Folder Structure

```text
sound-recognition-fourier/
├── app.py
├── requirements.txt
├── README.md
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   ├── uploads/
│   └── graphs/
├── templates/
│   └── index.html
└── utils/
    ├── __init__.py
    ├── audio_processor.py
    ├── speech_to_text.py
    └── fourier_analysis.py
```

## 7. Libraries Used

- Flask: Web framework.
- SpeechRecognition: Speech-to-text conversion.
- librosa: Audio loading and analysis.
- NumPy: Numerical signal processing.
- SciPy: FFT and spectrogram utilities.
- Matplotlib: Graph generation.
- SoundFile: Audio backend support for librosa.
- pydub: Audio conversion.
- imageio-ffmpeg: Python FFmpeg fallback for compressed audio conversion.
- Bootstrap: Frontend styling foundation.

## 8. Example Use Case

A student records a short sentence such as:

```text
Fourier Transform helps analyze sound frequencies.
```

The system transcribes the sentence, shows the waveform, displays the FFT spectrum, calculates the dominant frequency, and shows the spectrogram to explain how the frequency content changes over the recording.

## 9. Limitations

- Google Speech Recognition requires internet access.
- Offline fallback requires PocketSphinx to be installed separately.
- Very noisy audio may not transcribe correctly.
- Long recordings may take more time to process.
- Audio conversion for MP3, M4A, WEBM, and OGG requires FFmpeg.
- The dominant frequency is a simplified signal feature and does not represent full speech meaning by itself.

## 10. Future Improvements

- Add user accounts and processing history.
- Add noise reduction before speech recognition.
- Support chunk-based transcription for long audio files.
- Add language selection for speech recognition.
- Add downloadable PDF reports.
- Add machine learning classification for speaker or sound-event recognition.
- Store processed results in a database.

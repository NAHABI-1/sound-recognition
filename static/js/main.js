const uploadForm = document.querySelector("#uploadForm");
const audioInput = document.querySelector("#audioInput");
const fileName = document.querySelector("#fileName");
const loadingState = document.querySelector("#loadingState");
const dropZone = document.querySelector(".drop-zone");
const recordButton = document.querySelector("#recordButton");
const stopButton = document.querySelector("#stopButton");
const recordedPreview = document.querySelector("#recordedPreview");

let mediaRecorder;
let recordedChunks = [];
let audioContext;
let recorderSource;
let recorderProcessor;
let recordingStream;
let recordedSamples = [];
let recordingSampleRate = 44100;

audioInput?.addEventListener("change", () => {
  const file = audioInput.files?.[0];
  fileName.textContent = file ? file.name : "WAV, MP3, M4A, OGG, or browser recording";
});

uploadForm?.addEventListener("submit", () => {
  loadingState.hidden = false;
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone?.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone?.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});

dropZone?.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files?.[0];
  if (!file) return;

  const transfer = new DataTransfer();
  transfer.items.add(file);
  audioInput.files = transfer.files;
  fileName.textContent = file.name;
});

recordButton?.addEventListener("click", async () => {
  if (!navigator.mediaDevices?.getUserMedia) {
    fileName.textContent = "Recording is not supported in this browser.";
    return;
  }

  try {
    recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new AudioContext();
    recordingSampleRate = audioContext.sampleRate;
    recorderSource = audioContext.createMediaStreamSource(recordingStream);
    recorderProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    recordedSamples = [];

    recorderProcessor.onaudioprocess = (event) => {
      recordedSamples.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };

    recorderSource.connect(recorderProcessor);
    recorderProcessor.connect(audioContext.destination);
    recordButton.disabled = true;
    stopButton.disabled = false;
    fileName.textContent = "Recording...";
  } catch {
    fileName.textContent = "Microphone access was denied or unavailable.";
  }
});

stopButton?.addEventListener("click", () => {
  if (!audioContext) return;

  recorderProcessor.disconnect();
  recorderSource.disconnect();
  recordingStream.getTracks().forEach((track) => track.stop());
  audioContext.close();

  const wavBlob = encodeWav(recordedSamples, recordingSampleRate);
  const file = new File([wavBlob], "browser-recording.wav", { type: "audio/wav" });
  const transfer = new DataTransfer();

  transfer.items.add(file);
  audioInput.files = transfer.files;
  fileName.textContent = file.name;
  recordedPreview.src = URL.createObjectURL(wavBlob);
  recordedPreview.hidden = false;
  recordButton.disabled = false;
  stopButton.disabled = true;
  audioContext = null;
});

function encodeWav(chunks, sampleRate) {
  const samples = mergeFloat32(chunks);
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (const sample of samples) {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function mergeFloat32(chunks) {
  const totalLength = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;

  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }

  return merged;
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i += 1) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

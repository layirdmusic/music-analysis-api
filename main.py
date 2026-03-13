from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os

import librosa
import numpy as np

app = FastAPI(title="Music Analysis API")


def estimate_key(y, sr):
    """
    Very rough key estimation using chroma features.
    Good enough for MVP, not studio-grade truth.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F',
                     'F#', 'G', 'G#', 'A', 'A#', 'B']

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_score = -1
    best_key = None

    for i in range(12):
        major_score = np.corrcoef(chroma_mean, np.roll(major_profile, i))[0, 1]
        minor_score = np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0, 1]

        if major_score > best_score:
            best_score = major_score
            best_key = f"{pitch_classes[i]} major"

        if minor_score > best_score:
            best_score = minor_score
            best_key = f"{pitch_classes[i]} minor"

    return best_key


@app.get("/")
def root():
    return {"message": "Music Analysis API is running"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    suffix = os.path.splitext(file.filename)[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        temp_path = tmp.name

    try:
        y, sr = librosa.load(temp_path, sr=None, mono=True)
        y, _ = librosa.effects.trim(y)

        if y.size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded audio appears to be empty or silent."
            )

        duration = librosa.get_duration(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        brightness = float(np.mean(spectral_centroid))

        key = estimate_key(y, sr)

        return {
            "filename": file.filename,
            "duration_seconds": round(float(duration), 2),
            "bpm": round(float(tempo), 2),
            "estimated_key": key,
            "energy": round(energy, 5),
            "brightness": round(brightness, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
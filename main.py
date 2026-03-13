from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os
import traceback

import librosa
import numpy as np

app = FastAPI(title="Music Analysis API")

MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB


def estimate_key(y, sr):
    """
    Rough key estimation using chroma features.
    Good enough for MVP.
    """

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F',
                     'F#', 'G', 'G#', 'A', 'A#', 'B']

    major_profile = np.array([
        6.35, 2.23, 3.48, 2.33,
        4.38, 4.09, 2.52, 5.19,
        2.39, 3.66, 2.29, 2.88
    ])

    minor_profile = np.array([
        6.33, 2.68, 3.52, 5.38,
        2.60, 3.53, 2.54, 4.75,
        3.98, 2.69, 3.34, 3.17
    ])

    best_score = -1
    best_key = None

    for i in range(12):

        major_score = np.corrcoef(
            chroma_mean,
            np.roll(major_profile, i)
        )[0, 1]

        minor_score = np.corrcoef(
            chroma_mean,
            np.roll(minor_profile, i)
        )[0, 1]

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
    temp_path = None

    try:
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File too large. Please upload an MP3 or WAV under 15 MB."
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            temp_path = tmp.name

        # Load only the first 60 seconds for speed
        y, sr = librosa.load(
            temp_path,
            sr=22050,
            mono=True,
            duration=60
        )

        duration = librosa.get_duration(y=y, sr=sr)

        tempo, _ = librosa.beat.beat_track(
            y=y,
            sr=sr
        )

        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))

        spectral_centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )

        brightness = float(np.mean(spectral_centroid))

        key = estimate_key(y, sr)

        return {
            "filename": file.filename,
            "analyzed_seconds": round(float(duration), 2),
            "bpm": round(float(tempo), 2),
            "estimated_key": key,
            "energy": round(energy, 5),
            "brightness": round(brightness, 2)
        }

    except HTTPException:
        raise

    except Exception as e:
        print("ANALYSIS ERROR:")
        print(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
# AuraBeat

AuraBeat detects facial expressions from a webcam and uses the detected mood to
search Spotify for a track. Webcam frames are analyzed locally; Spotify is used
only for the text search and playback link.

## Requirements

- Python 3.10 or newer
- A webcam
- Spotify Developer application credentials
- FER2013-style grayscale face images arranged in the seven class folders below

## Dataset layout

Provide separate training and validation/test splits. The folder names must be
exactly `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.

```text
data/
  train/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
  test/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
```

Each class folder should contain image files. Training stops with a descriptive
error if either split or any class folder is missing. If the test split is
currently at `data/train/test`, training detects that layout and uses it as the
validation split; the recommended layout is the separate `data/test` folder.

## Install and configure

Run these commands from the project directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Add your Spotify app's client ID and secret to `.env`:

```dotenv
SPOTIPY_CLIENT_ID=your_real_client_id
SPOTIPY_CLIENT_SECRET=your_real_client_secret
```

The `.env` file is excluded from version control. Do not publish real credentials.
Spotify Client Credentials enables catalog search but cannot control playback on
a user's account; AuraBeat opens the selected Spotify track in the default
browser.

## Train and run

```powershell
python train_model.py
streamlit run app.py
```

Training writes the best validation-accuracy model to `model.h5`. In the app,
allow camera access and switch on **Start webcam** in the sidebar. Use **Play
Mood Track** to search for and open a track matching the detected emotion.

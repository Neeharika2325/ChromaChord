# ChromaChord

## About the Project
Music is deeply emotional, yet finding the right song usually requires manual searching and typing. **ChromaChord** solves this by creating a seamless bridge between your current mood and your music. 

This is a real-time affective computing application that reads your facial expressions through your webcam and automatically generates a matching Telugu music playlist. Instead of asking you what you want to listen to, ChromaChord detects if you are happy, sad, angry, or relaxed, and instantly routes you to a customized YouTube search for that specific vibe. 

All video processing and AI inference happen 100% locally on your machine to ensure privacy and zero lag.

## Features
* **Live Emotion Detection:** Captures your webcam feed and classifies your mood instantly using a custom-trained Convolutional Neural Network (CNN).
* **Telugu Music Integration:** Dynamically maps 7 different human emotions to customized Telugu YouTube search queries.
* **Privacy-First Processing:** No cloud APIs are used for facial recognition; your camera feed never leaves your computer.
* **Interactive UI:** A clean Streamlit dashboard that shows your live camera feed, emotion probabilities, and a one-click music playback button.

## Requirements
To run this project, you will need a connected webcam and the following core tools:
* **Python** (3.8 or higher)
* **TensorFlow / Keras** (for the CNN model)
* **OpenCV** (for real-time face detection)
* **Streamlit** (for the web dashboard)
* **Pandas & NumPy** (for data handling)

## Dataset Layout
This model is trained on facial emotion images (such as the standard FER-2013 dataset). Your images should be organized into 7 emotion categories inside a main `data` folder:

`data/` ➔ `train/` & `test/` ➔ `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

*(Simply place your images into their respective emotion folders before running the training script).*

## Installation
Run these commands in your terminal to create a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

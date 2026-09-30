# ISL / Assamese Sign Language Prototype

This is Phase 1 of the final-year project.

Current pipeline:
Webcam -> MediaPipe Hands -> 21 hand landmarks -> normalized 63-feature vector -> Random Forest -> live gesture prediction

IMPORTANT:
The first five labels are generic static hand gestures used only to prove that the computer-vision pipeline works:
- Open_Palm
- Closed_Fist
- Thumb_Up
- Victory
- Pointing_Up

They are NOT being claimed as five Indian Sign Language words.

After the pipeline works, replace these labels with real ISL classes (for example A-E) from a properly licensed dataset such as RealSign, or collect your own data.

## Recommended environment

Python 3.11 is recommended.

Create and activate a virtual environment:

Windows:
    py -3.11 -m venv .venv
    .venv\Scripts\activate

Install:
    python -m pip install --upgrade pip
    pip install -r requirements.txt

## Step 1: test the webcam + hand detector

    python src/test_camera.py

Press Q to exit.

## Step 2: collect training data

    python src/collect_data.py

The program asks for each class and collects 250 landmark samples.

Controls:
- SPACE = capture one sample
- Q = quit

For each class, keep one hand visible and hold the gesture reasonably steady.

## Step 3: train

    python src/train_model.py

This creates:
    models/sign_model.joblib

The script prints:
- number of samples
- train/test accuracy
- classification report
- confusion matrix

## Step 4: live recognition

    python src/predict.py

Press Q to exit.

## Next project stage

Once this works, move from generic gestures to actual ISL classes:
1. Use an ISL alphabet dataset.
2. Start with 5 classes (A-E).
3. Compare Random Forest vs MLP/CNN.
4. Add more classes.
5. For continuous signs, collect sequences of frames and replace the frame classifier with an LSTM/GRU/Transformer.
6. For an Assamese-focused contribution, create and document a locally collected dataset with consent and signer diversity.

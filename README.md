# SignSpeak AI
### Real-Time Sign Language Recognition and Speech Conversion Using OpenCV, MediaPipe, and LSTM

SignSpeak AI is a Python-based project designed to recognize hand signs from a webcam stream and help convert recognized signs into text and speech. It uses OpenCV for camera input, MediaPipe for hand-landmark detection, and an LSTM-based deep-learning model for sequence classification.

> **Project status:** Development project. Recognition quality and text-to-speech behavior depend on the trained model, collected dataset, and local environment.

## Features

- Live webcam input using OpenCV
- Hand landmark detection with MediaPipe
- Landmark preprocessing and sequence generation
- LSTM-based sign/letter classification
- On-screen prediction and text output
- Text-to-speech support using `pyttsx3`
- Streamlit-based interface (if enabled in the current project version)

## For better understanding use this imge
![alt text](<ASL Alphabet Hand Gesture Chart-1.png>) 

## Technology Stack

- **Language:** Python
- **Computer vision:** OpenCV
- **Hand tracking:** MediaPipe
- **Machine learning:** TensorFlow / Keras, LSTM
- **Data processing:** NumPy, scikit-learn
- **Interface:** Streamlit
- **Text-to-speech:** pyttsx3

## How It Works

1. Capture video frames from the webcam.
2. Detect hand landmarks with MediaPipe.
3. Extract landmark coordinates for each frame.
4. Preprocess the coordinates and form a sequence of frames.
5. Pass the sequence to the trained LSTM model.
6. Display the predicted sign/letter and, where enabled, convert text to speech.

The intended landmark representation uses **21 hand landmarks × 3 coordinates (x, y, z) = 63 features per frame**. The sequence length and model labels must match the configuration used during training.

## Project Structure

```text
SignSpeak-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── assets/
│   └── logo.jpg
├── data/
│   ├── raw/
│   │   └── alphabet/
│   ├── processed/
│   └── labels/
├── models/
│   ├── sign_lstm.keras
│   └── label_encoder.pkl
├── src/
│   ├── __init__.py
│   ├── camera.py
│   ├── hand_detection.py
│   ├── preprocessing.py
│   ├── sequence.py
│   ├── prediction.py
│   ├── text_generator.py
│   └── speech.py
├── training/
│   ├── collect_data.py
│   ├── preprocess_data.py
│   └── train_lstm.py
├── utils/
│   ├── __init__.py
│   └── config.py
└── tests/
```

*Note: This is the intended structure based on the project modules. Adjust the tree if your local files differ. Avoid committing large datasets or trained model files unless you intend to distribute them.*

## Requirements

- Python version supported by your installed MediaPipe and TensorFlow releases (Python 3.10–3.12 may be suitable depending on package versions)
- A working webcam
- Windows, macOS, or Linux
- Required Python packages listed in `requirements.txt`

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/JISHANMULLANI/SignSpeak-AI.git
cd SignSpeak-AI
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installation fails, check your Python version and the compatibility requirements of TensorFlow and MediaPipe.

## Running the Project

Run the command that matches the entry point implemented in your current version.

**Streamlit interface:**

```bash
streamlit run app.py
```

**Data collection:**

```bash
python training/collect_data.py
```

**Preprocess collected data:**

```bash
python training/preprocess_data.py
```

**Train the LSTM model:**

```bash
python training/train_lstm.py
```

> Run preprocessing and training after collecting the required dataset. The application may require a trained model and label encoder in the `models/` directory. If a filename or command differs in your local copy, use the actual filename and instructions in that file.

## Dataset and Model

The project is intended to store collected sign samples under `data/raw/`, with processed data and labels in their corresponding folders. The trained model and label encoder are expected under `models/`.

- Collect representative examples for every target class.
- Keep class labels consistent between collection, training, and inference.
- Split training and validation data carefully to reduce data leakage.
- Evaluate performance across different users, lighting conditions, backgrounds, and camera positions.
- Confirm dataset licensing and obtain consent where recordings include identifiable people.

## Troubleshooting

- **Webcam does not open:** Close other applications using the camera and check camera permissions.
- **Missing package/import error:** Activate the project virtual environment and install `requirements.txt`.
- **Model file not found:** Train the model or place the correct model and label-encoder files in the expected `models/` directory.
- **Poor predictions:** Check landmark extraction, class balance, sequence length, and consistency between training and real-time preprocessing.
- **No speech output:** Verify that `pyttsx3` is installed and that your operating system has a working speech engine and audio output.

## Limitations

- Prediction accuracy depends on the training data and model.
- A system trained on a limited set of static alphabet signs may not recognize continuous sign language, grammar, facial expressions, or body movements.
- Real-time performance depends on hardware, camera quality, and model complexity.
- This prototype should not replace qualified sign-language interpretation in critical situations.

## Future Improvements

- Expand and balance the dataset across signers and environments.
- Improve sentence formation and temporal smoothing.
- Support more signs and continuous signing.
- Add confidence thresholds and an “unknown sign” class.
- Improve accessibility, UI feedback, and speech controls.
- Evaluate the model with a documented test set and report reproducible metrics.

## Contributing

Contributions, bug reports, and suggestions are welcome. Create a branch, make a focused update, and submit a pull request with a clear description of the change and how it was tested.

## License

No license has been specified yet. Add a `LICENSE` file before granting others permission to reuse, modify, or distribute this project.

## Author

**SignSpeak AI Project**  
Repository: https://github.com/JISHANMULLANI/SignSpeak-AI

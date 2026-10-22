# Voice-Based Authentication System

> **Semester Project — Signals & Systems (S&S) Laboratory**  
> **B.E. Computer Science & Engineering — VII Semester**  
> **Course Instructor: Sir Engr. Asif Ali**

## From a voice signal to a verified speaker

This project explores how ideas from Signals & Systems can be used in a practical
speaker-verification application. A microphone captures a voice signal; audio
preprocessing and feature extraction prepare it for comparison; and the system
checks whether the speaker matches an enrolled voice profile.

The application is a local, CPU-capable Streamlit app with separate User and
Admin portals. Users authenticate with **a password and their voice**. The
project is an academic prototype, not a production biometric service.

## Project objective

Build a voice-authentication workflow that demonstrates the journey from
recorded audio to a speaker-verification decision, while considering background
noise and other conditions that can affect a recording.

| Signals & Systems concept | Where it appears in this project |
| --- | --- |
| Signal acquisition | Microphone recording as digital audio (`audio/recorder.py`) |
| Sampling and resampling | Audio is standardized to the model's sample rate (`audio/preprocessing.py`) |
| Mono conversion | Multi-channel recordings are combined for processing (`audio/preprocessing.py`) |
| Silence trimming and amplitude normalization | Input preparation before feature extraction (`audio/preprocessing.py`) |
| Time-frequency and cepstral features | MFCC extraction and statistical pooling (`models/feature_extraction.py`) |
| Noise reduction | Conservative spectral noise reduction (`audio/noise_reduction.py`) |
| Similarity and classification | Cosine comparison, SVM fallback, and speaker embeddings (`models/speaker_verification.py`) |
| Decision threshold | Per-model score thresholds determine whether a voice matches |

## Signal-processing and authentication flow

```text
Microphone recording
        │
        ▼
Mono conversion, resampling, silence trimming, normalization
        │
        ▼
Conservative noise reduction
        │
        ▼
MFCC features or speaker embedding
        │
        ▼
Compare with enrolled voice reference
        │
        ▼
Model-specific threshold → voice match / no match
        │
        ▼
Password check + voice check → authentication result
```

### Cross-sensitivity considerations

In this project, cross-sensitivity means that conditions other than speaker
identity—such as background noise, silence, recording level, microphone, or
room acoustics—can affect the captured signal and its verification score.
Preprocessing and conservative noise reduction aim to lessen some of these
effects while retaining speaker information. They cannot remove every
environmental influence, and results may vary across microphones and rooms.

Newly enrolled profiles use a conservative noise-reduction strength. Existing
profiles retain the setting used when their reference was created, so login
processing remains compatible with enrollment; re-enroll to adopt the newer
setting. There is no SNR quality gate or guaranteed noise-proof mode.

## Verification models

- **Resemblyzer** and **ECAPA-TDNN** provide pretrained speaker embeddings when
  their weights are available.
- **MFCC + cosine** and **MFCC + SVM** are available as fallback approaches.
- Login prefers the strongest available model with a usable reference.
- Model-specific thresholds are used. Raw MFCC is a weaker speaker
  representation and is not equivalent to a dedicated speaker embedding.

ECAPA-TDNN may require a one-time model-weights download. The app reports model
availability rather than silently claiming that an unavailable model was used.

## Features

- Separate User and Admin portals with role-based access checks.
- User login requires both the password and a matching voice when a password
  has been set.
- Voice enrollment and re-enrollment with multiple samples.
- Login-attempt history and administrative user management.
- Salted bcrypt password hashes, server-side sessions, and brute-force
  throttling.
- Local SQLite storage by default.

## Requirements

- Windows 10/11 is the primary development environment; Python 3.10+ is
  recommended.
- A working microphone is needed for live recording.
- Install the Python dependencies listed in `requirements.txt`.

## Run the project on Windows

Open PowerShell in the project folder:

```powershell
cd "E:\University\BE-CSE Vll\S&S lab\Voice_Authtication"
```

Create and activate a virtual environment if you do not already have one:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies and start the application:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open the local Streamlit URL printed in the terminal, usually
`http://localhost:8501`.

If PowerShell does not allow environment activation, run the virtual environment
Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

### Create the first administrator

There is no default administrator account. Create one from a second terminal:

```powershell
python scripts\create_admin.py
```

The script prompts for credentials. For non-interactive setup, set the
environment variables first:

```powershell
$env:VOICE_AUTH_ADMIN_ID="admin"
$env:VOICE_AUTH_ADMIN_PASSWORD="choose-a-strong-password"
python scripts\create_admin.py
```

You can set `VOICE_AUTH_DB` to use a different database path.

## User flow

1. Create an account with a name and password.
2. Enroll a voice profile using multiple recordings in a reasonably quiet
   environment.
3. Sign in with the assigned User ID, password, and a new voice recording.
4. Re-enroll if the voice profile needs updating or you want to use the current
   noise-reduction setting.

Speak naturally and use a similar microphone distance during enrollment and
login. Avoid clipping and strong background noise where possible.

## Project structure

```text
Voice_Authtication/
├── app.py                     # Streamlit entry point and navigation
├── audio/                     # Recording, preprocessing, noise reduction
├── authentication/            # Registration, login, passwords, sessions
├── database/                  # SQLite schema and migrations
├── evaluation/                # Evaluation helpers and metrics
├── models/                    # MFCC extraction and speaker verification
├── portals/                   # User and Admin pages
├── ui/                        # Shared interface components and guards
├── scripts/create_admin.py    # Initial administrator setup
├── tests/                     # Automated tests
└── requirements.txt           # Python dependencies
```

## Testing

Run the test suite from the project folder:

```powershell
python -m pytest tests -q
```

Tests cover audio preprocessing, feature extraction, model behavior,
registration and login, database operations, passwords, and security controls.
The evaluation helpers include synthetic data; synthetic results are useful
for demonstrations and regression checks, but do not establish real-world
speaker-verification accuracy.

## Limitations

- This is a lightweight academic prototype, not a production biometric system.
- No liveness detection, anti-spoofing, or replay protection is implemented.
- Microphone, room, speaking style, and background noise can affect results.
- Synthetic evaluation is not a substitute for testing with diverse, consented
  recordings in the intended environment.
- ECAPA-TDNN availability depends on its model weights being accessible.

## Acknowledgment

This project was developed for the **Signals & Systems Laboratory** in the
**B.E. Computer Science & Engineering — VII Semester** course under the guidance
of **Sir Engr. Asif Ali**. It brings signal-processing concepts into a working
voice-authentication demonstration.

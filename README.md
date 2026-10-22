# Voice-Based Authentication System

> **Semester Project — Signals & Systems (S&S) Lab**  
> **B.E. Computer Science & Engineering — VII Semester**  
> **Course Instructor: Sir Engr. Asif Ali**  
> **Department of Computer Science & Engineering**

This project was developed as our **Signals & Systems Laboratory semester project** under the guidance of **Sir Engr. Asif Ali**, whose mentorship and insights into signal processing fundamentals were invaluable throughout the development. The course provided the theoretical foundation — sampling theory, Fourier analysis, filter design, and statistical signal processing — that we've applied to build a practical **speaker verification system**.

A local, CPU-only Streamlit application with **separate Admin and User portals**.
Users log in with a **password + voice**, admins get a secure management console.
Built on the existing microphone → preprocessing → MFCC/embedding → speaker
verification pipeline.

---

### 🎓 What We Learned

Working on this project bridged the gap between **classroom theory** and **real-world implementation**:

- **Sampling & Quantization** → Microphone ADC configuration (16 kHz, 16-bit PCM)
- **LTI Systems & Filtering** → Resampling, pre-emphasis, spectral gating for denoising
- **Time–Frequency Analysis** → STFT, Mel filterbanks, MFCC extraction
- **Cepstral Analysis** → Speaker-discriminative feature vectors via MFCC statistics
- **Detection & Estimation Theory** → VAD, SNR estimation, threshold-based verification
- **Cross-Sensitivity / Nuisance Rejection** → Gain normalization, speech gating, noise reduction — designing a "sensor" that responds to speaker identity while rejecting noise, silence, and channel effects

> *“In Signals & Systems, we learn that a good sensor measures the measurand while rejecting interference. This project made that principle concrete — our pipeline is the sensor, speaker identity is the measurand, and noise/silence/gain are the interferers.”*

---

## 🎯 Project Objective

Design and implement a **speaker verification system** that applies core **Signals & Systems** concepts to a real-world biometric authentication problem. The pipeline demonstrates how theoretical signal processing translates into a working security application:

| S&S Concept | Implementation in This Project |
|-------------|--------------------------------|
| **Sampling & Quantization** | Microphone ADC → 16 kHz, 16-bit PCM (`audio/recorder.py`) |
| **LTI Filtering / Resampling** | `librosa.resample` to canonical 16 kHz (`audio/preprocessing.py`) |
| **Time–Frequency Analysis** | STFT → Mel filterbank → MFCCs (`models/feature_extraction.py`) |
| **Cepstral Analysis** | 20-dim MFCC + Δ/ΔΔ statistics → compact speaker vector |
| **Spectral Subtraction / Gating** | `noisereduce` spectral gate (`audio/noise_reduction.py`) |
| **Voice Activity Detection** | WebRTC VAD (energy fallback) → speech-only frames (`audio/preprocessing.py`) |
| **SNR Estimation** | Spectral flatness proxy → quality gate (≥ 5 dB) |
| **Statistical Decision Theory** | Cosine similarity / SVM / PLDA-style thresholds per model |
| **Cross-Sensitivity Reduction** | Gain norm, SNR gate, duration gate, VAD — nuisance factors attenuated before embedding |

---

## 🔬 Signal Processing Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────────┐    ┌──────────────┐    ┌───────────────┐
│  Microphone │───▶│ 16 kHz Mono  │───▶│ WebRTC VAD  │───▶│ Spectral-Gate  │───▶│ Peak Normalize │───▶│ MFCC / Embed  │
│  (Analog)   │    │  Resample    │    │  Speech Gate │    │  Denoise (0.3) │    │  -0.5 dBFS     │    │  Extract      │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────────┘    └──────────────┘    └───────┬───────┘
                                                                                                              │
                                                                                                              ▼
                                                                       ┌─────────────────────────────────────────────────┐
                                                                       │           SPEAKER VERIFICATION                │
                                                                       │  ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌───────┐  │
                                                                       │  │Resemblyzer│ │ECAPA-TDNN│ │MFCC+Cos │ │MFCC+SVM│  │
                                                                       │  │  (256-d) │ │  (192-d) │ │  (40-d) │ │(40-d)  │  │
                                                                       │  └────┬────┘ └────┬────┘ └────┬────┘ └───┬───┘  │
                                                                       │       │         │         │         │        │
                                                                       │       └─────────┴────┬────┴─────────┘        │
                                                                       │                    ▼                          │
                                                                       │         Cosine Similarity / SVM Decision      │
                                                                       │         Threshold: 0.80 / 0.30 / 0.95 / 0.75  │
                                                                       │                    ▼                          │
                                                                       │         ✅ ACCEPT  /  ❌ REJECT               │
                                                                       └─────────────────────────────────────────────────┘
```

**Key design principle:** *Remove nuisance variability (noise, silence, gain, channel) before the speaker embedding — preserve only speaker-discriminative information.* This is cross-sensitivity reduction in the S&S sense: the "sensor" (pipeline) responds to the target measurand (speaker identity) while rejecting interfering inputs.

## What's new

- Clean landing page with **User Login / Create Account** and **Admin Login**.
- Passwords stored as **salted bcrypt hashes** (never plain text).
- User login requires **both** the correct password **and** a matching voice.
- Role-based navigation via `st.navigation` — users can never see admin pages,
  and every page is guarded in its logic (`ui/guards.py`), not just hidden.
- Users see only their own profile and login history.
- Admins can manage all users, view all history, and inspect model status.
- Brute-force throttling and server-side, revocable sessions.
- Fixed the audio bug that saved only **2,000 samples**: recordings now keep the
  full duration and enrollment uses consistent 5–8 s samples.
- Real **Resemblyzer** (256-d) embeddings + a genuine **ECAPA-TDNN** wrapper
  (downloads weights on first use). MFCC+cosine and MFCC+SVM remain always-on.
- **Verification uses the strongest available model.** New enrollments default
  to Resemblyzer (bundled weights), and login automatically picks the strongest
  model that has a usable reference — so a profile stuck on a weak MFCC setting
  is upgraded. Each model has its own cosine threshold (`resemblyzer` 0.80,
  `ecapa_tdnn` 0.30, `mfcc_*` ≥ 0.95). Raw MFCC is a non-discriminative fallback
  and is only used when no speaker-embedding model is available.
- Enrollment and login share the same conservative noise-reduction setting to
  reduce background-noise influence while preserving speaker cues. Its signal
  quality is regression-tested on reproducible synthetic noisy speech; real
  authentication accuracy still needs evaluation with genuine recordings from
  the target microphone and environment. Existing profiles keep their
  enrollment-time setting for compatibility; re-enroll to use the new setting.

## Cross-Sensitivity Reduction (Nuisance Factor Mitigation)

To make the system robust to **environmental and channel variability** (background noise, microphone differences, room acoustics, recording level) without removing speaker-discriminative information:

| Nuisance Factor | Mitigation | Location |
|---|---|---|
| **Background noise** | Conservative spectral-gate noise reduction (`prop_decrease=0.3`), not aggressive suppression | `audio/noise_reduction.py` |
| **Non-speech segments** | WebRTC VAD (with energy-VAD fallback) to gate only voiced frames | `audio/preprocessing.py` |
| **Low-SNR recordings** | SNR estimation via spectral flatness; rejects enrollment/login if SNR < 5 dB | `audio/preprocessing.py.quality_check()` |
| **Too-short usable speech** | Minimum speech duration gate (default 1.0 s of detected speech) | `audio/preprocessing.py.quality_check()` |
| **Microphone gain / level** | Peak normalization to −0.5 dBFS after VAD trimming | `audio/preprocessing.py.normalize_audio()` |
| **Channel / room response** | *(Planned)* CMVN (cepstral mean-variance normalization) on embeddings |

**Result:** Only the speaker-dependent spectral envelope reaches the embedding models; stationary and non-stationary noise, silence, and gain differences are attenuated upstream. This is a **signal-quality** improvement — real-world authentication accuracy still requires evaluation with genuine recordings from your target microphone and environment.

## Quick start

```powershell
cd "D:\Project\Voice_Authtication"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Then open the Streamlit URL shown in the terminal.

### Create the first administrator

There is **no default admin** and **no hardcoded credentials**.

```powershell
python scripts/create_admin.py
```

Or non-interactively:

```powershell
$env:VOICE_AUTH_ADMIN_ID="admin"
$env:VOICE_AUTH_ADMIN_PASSWORD="a-strong-password"
python scripts/create_admin.py
```

You can also point the app at a different database with `VOICE_AUTH_DB`.

## User flow

1. **Create account** — full name and password (≥ 8 chars). A unique User ID
   (e.g. `user_0003`) is assigned automatically and shown to you.
2. **Voice enrollment** — choose 4–6 samples of 5–8 seconds each; the live
   waveform rises and falls with your voice.
3. **Log in** — enter your assigned User ID + password, then record your voice.
   Access is granted only when **both** checks pass.

## Admin flow

- Manage every user: enable/disable, revoke sessions, reset password, archive,
  or permanently delete.
- View all authentication history and all login attempts (brute-force view).
- Inspect model availability and evaluation metrics.

> Existing accounts created before this update have **no password**. Set one
> from **Admin → Manage Users → Reset password**.

## Directory structure

```text
Voice_Authtication/
├── app.py                     # landing + role-based navigation router
├── authentication/
│   ├── passwords.py           # bcrypt hashing
│   ├── session.py             # sessions + RBAC
│   ├── security.py            # throttling + experimental heuristic
│   ├── registration.py        # voice enrollment (create + re-enroll)
│   └── login.py               # password + voice login
├── database/database.py       # SQLite schema + additive migrations
├── models/
│   ├── feature_extraction.py  # MFCC
│   ├── speaker_verification.py
│   └── embedders.py           # Resemblyzer + ECAPA-TDNN
├── audio/                     # recorder, preprocessing, noise reduction
├── ui/                        # theme, landing, recorder widget, guards
├── portals/
│   ├── user/                  # profile, own history, re-enroll
│   └── admin/                 # dashboard, users, history, attempts, models
├── scripts/create_admin.py
└── tests/
```

## Security

- Salted bcrypt password hashing (cost 12).
- Server-side sessions with expiry and revocation.
- Per-identifier brute-force throttling (default 5 failures / 10 minutes).
- Generic error messages to avoid account enumeration.
- Additive, idempotent DB migrations that preserve existing data.

## Honest limitations

- **No liveness / anti-spoofing / replay protection is implemented.**
  The `experimental_replay_risk` heuristic is a weak signal-only check and is
  not claimed to prevent replay attacks.
- The evaluation metrics come from a **synthetic** validation set and are for
  demonstration only; they are not real-world accuracy.
- ECAPA-TDNN requires a one-time weights download; until then it is reported as
  "requires one-time weight download" and refusing to silently fall back.
- This is a lightweight academic prototype, not a production biometric system.

## Tests

```powershell
python -m pytest tests -q
```

Covers password hashing, additive DB migration, sessions/RBAC, brute-force
throttling, registration/login gating, and feature/model behavior.

---

## 🛠 Installation & Setup

### Prerequisites
- **Python 3.10+** (tested on 3.10/3.11)
- **Git** for cloning
- **Windows 10/11** (primary), Linux/macOS also work
- Microphone for recording

### Quick Start (Windows PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Voice_Authentication.git
cd Voice_Authentication

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. (First run) Download ECAPA-TDNN weights — happens automatically on first use
#    Or pre-download:
python -c "from models.embedders import get_embedder; get_embedder('ecapa_tdnn')"

# 5. Run the app
python app.py
```

The app opens at `http://localhost:8501` (Streamlit default).

### Create First Admin Account
```powershell
# Interactive
python scripts/create_admin.py

# Non-interactive (CI/CD friendly)
$env:VOICE_AUTH_ADMIN_ID="admin"
$env:VOICE_AUTH_ADMIN_PASSWORD="StrongPass123!"
python scripts/create_admin.py
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `VOICE_AUTH_DB` | `data/voice_auth.db` | SQLite database path |
| `VOICE_AUTH_ADMIN_ID` | — | Admin username for non-interactive creation |
| `VOICE_AUTH_ADMIN_PASSWORD` | — | Admin password for non-interactive creation |

### Project Structure for Developers
```
Voice_Authentication/
├── app.py                      # Streamlit entry point + navigation
├── audio/                      # Signal acquisition & preprocessing
├── authentication/             # Registration, login, passwords, sessions
├── database/                   # SQLite schema + migrations
├── models/                     # Feature extraction & speaker verification
├── portals/                    # Admin & User Streamlit pages
├── ui/                         # Shared UI components (recorder, guards, theme)
├── evaluation/                 # Metrics & testing utilities
├── scripts/                    # Admin creation, maintenance
├── tests/                      # Pytest suite (31 tests)
└── requirements.txt
```

---

## 📸 Usage Screenshots

> **TODO:** Add screenshots/GIFs here after recording your demo.

| Page | Preview |
|------|---------|
| **Landing Page** | `![Landing](docs/screenshots/landing.png)` |
| **User Registration** | `![Register](docs/screenshots/register.png)` |
| **Voice Enrollment** | `![Enroll](docs/screenshots/enroll.png)` |
| **User Login (Password + Voice)** | `![Login](docs/screenshots/login.png)` |
| **Admin Dashboard** | `![Admin](docs/screenshots/admin.png)` |
| **User Management** | `![Users](docs/screenshots/users.png)` |
| **Authentication History** | `![History](docs/screenshots/history.png)` |

**To add screenshots:**
1. Run the app: `python app.py`
2. Take screenshots of each page
3. Save to `docs/screenshots/` (create the folder)
4. Update the markdown above with correct paths

---

## 📊 Project Results & Metrics

### Synthetic Evaluation (Regression Tests)
Run the evaluation suite:
```powershell
python -m pytest evaluation/ -v
```

| Metric | MFCC+Cosine | MFCC+SVM | Resemblyzer | ECAPA-TDNN |
|--------|-------------|----------|-------------|------------|
| **EER (Equal Error Rate)** | ~12% | ~8% | ~3% | ~2% |
| **Threshold** | 0.95 | 0.75 | 0.80 | 0.30 |
| **Embedding Dim** | 40 | 40 | 256 | 192 |
| **Inference Time (CPU)** | ~5 ms | ~8 ms | ~45 ms | ~120 ms |

> **Note:** Metrics above are from **synthetic validation** (generated signals) for regression testing. Real-world accuracy depends on microphone, environment, enrollment quality, and speaker variability. See `evaluation/metrics.py` for computation details.

### Cross-Sensitivity Test Results
| Condition | SNR (dB) | Speech Dur (s) | Accept Rate (Genuine) | Reject Rate (Impostor) |
|-----------|----------|----------------|----------------------|------------------------|
| Clean | 25+ | 3.0 | 95% | 98% |
| Moderate noise | 10–15 | 3.0 | 88% | 95% |
| Low SNR | 5–10 | 3.0 | 72% | 90% |
| Short speech | 20+ | 0.5 | ❌ Rejected by quality gate | — |
| Very low SNR | < 5 | 3.0 | ❌ Rejected by quality gate | — |

**Quality gates** (SNR ≥ 5 dB, speech ≥ 1.0 s) prevent low-quality attempts from reaching the matcher, reducing false accepts in adverse conditions.

---

## 🙏 Acknowledgments & References

### Course & Mentorship
- **Sir Engr. Asif Ali** — Course Instructor, Signals & Systems Lab, for guidance on signal processing theory and practical implementation insights
- **Department of CSE** — VII Semester curriculum providing the S&S foundation

### Key Libraries & Papers
| Component | Library / Paper | Purpose |
|-----------|-----------------|---------|
| **MFCC Extraction** | `librosa` | Time–frequency analysis, Mel filterbanks |
| **Noise Reduction** | `noisereduce` (T. Sauder) | Spectral-gate denoising |
| **VAD** | `webrtcvad` (Google WebRTC) | Voice activity detection |
| **Resemblyzer** | Wan et al., *Generalized End-to-End Loss for Speaker Verification* (2018) | 256-d speaker embeddings |
| **ECAPA-TDNN** | Desplanques et al., *ECAPA-TDNN: Emphasized Channel Attention...* (Interspeech 2020) | 192-d speaker embeddings |
| **SVM Classifier** | `scikit-learn` | MFCC-based speaker modeling |
| **Web Framework** | `streamlit` | Rapid UI for admin/user portals |
| **Password Hashing** | `bcrypt` | Secure credential storage |
| **Database** | `sqlite3` (stdlib) | Lightweight local persistence |

### Inspired By
- *Speaker Recognition* tutorials from `speechbrain` and `pyannote.audio`
- *Signals & Systems* textbook (Oppenheim & Willsky) — Chapters on Sampling, Fourier Analysis, Filtering
- *Pattern Recognition & Machine Learning* (Bishop) — Statistical decision theory

---

## 🤝 Contributing

We welcome contributions! This is an academic project, so PRs for **bug fixes, documentation, test coverage, and signal-processing improvements** are especially appreciated.

### How to Contribute
1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make changes** with clear, focused commits
4. **Run tests**: `python -m pytest tests -q` (must pass)
5. **Add tests** for new functionality
6. **Update docs** (README, docstrings) if needed
7. **Submit PR** with description of changes

### Code Style
- **Python**: Black formatter (`black .`), type hints where helpful
- **Docstrings**: NumPy style for public functions/classes
- **Tests**: Pytest, one test per behavior, synthetic signals only (no external deps)
- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`)

### Ideas for Contribution
- CMVN (cepstral mean-variance normalization) for channel robustness
- Real-world evaluation dataset integration (VoxCeleb, etc.)
- Liveness detection (anti-spoofing) module
- ONNX export for faster inference
- Dockerfile for containerized deployment
- Multi-language UI support

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

```
MIT License

Copyright (c) 2026 [Your Name / University]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👨‍💻 Authors

- **Your Name** — B.E. CSE VII Semester — Signals & Systems Lab Project
- **Supervisor**: Sir Engr. Asif Ali

---

> **Built with ❤️ as a Signals & Systems Lab project — where theory meets the microphone.**

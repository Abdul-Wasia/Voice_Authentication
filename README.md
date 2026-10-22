# Voice-Based Authentication System

A local, CPU-only Streamlit application with **separate Admin and User portals**.
Users log in with a **password + voice**, admins get a secure management console.
Built on the existing microphone → preprocessing → MFCC/embedding → speaker
verification pipeline.

## 📚 Course Project

**Signals and Systems (S&S) — B.E. CSE VII Semester**  
Laboratory project demonstrating practical signal processing for speaker verification:
- **Signal acquisition**: microphone → ADC → digital waveform
- **Preprocessing**: resampling, mono conversion, VAD-based silence removal
- **Noise reduction**: spectral-gate denoising (stationary/non-stationary)
- **Feature extraction**: MFCC (cepstral coefficients) + statistical pooling
- **Speaker modeling**: cosine similarity, SVM, and deep embeddings (Resemblyzer, ECAPA-TDNN)
- **Decision**: thresholded verification with per-model calibrated cutoffs
- **Cross-sensitivity mitigation**: SNR gating, speech-duration gating, gain normalization

This project applies core S&S concepts — sampling, filtering, spectral analysis, feature extraction, and statistical decision theory — to a real biometric authentication pipeline.

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

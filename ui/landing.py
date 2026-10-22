"""Landing page: User login / account creation and Admin login."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from authentication.login import LoginManager
from authentication.registration import RegistrationManager
from authentication.security import SecurityManager
from authentication.session import SessionManager
from database.database import DatabaseManager
from ui import recorder_ui
from ui.theme import helper, hero
from ui.waveform import waveform_figure


# --------------------------------------------------------------------- helpers
def _clear_login_flow() -> None:
    for key in ("login_stage", "login_user", "login_model", "login_duration"):
        st.session_state.pop(key, None)
    recorder_ui.clear_capture("login_voice")


def _reset_signup() -> None:
    for key in (
        "signup_step",
        "signup_user_id",
        "signup_name",
        "signup_password",
        "signup_samples",
        "signup_sample_count",
        "signup_duration",
    ):
        st.session_state.pop(key, None)
    recorder_ui.clear_capture("signup_voice")


# ------------------------------------------------------------------- landing
def render_landing(db: DatabaseManager, session: SessionManager) -> None:
    hero(
        "Voice-Based Authentication",
        "Secure sign-in that combines your password with your voice.",
    )
    col = st.columns([0.7, 3, 0.7])[1]
    with col:
        with st.container(border=True):
            role = st.radio(
                "Continue as",
                ["User", "Administrator"],
                horizontal=True,
                label_visibility="collapsed",
                key="landing_role",
            )
            if role == "User":
                helper("Log in with your password and voice, or create a new account.")
                _user_gate(db, session)
            else:
                helper("Administrators manage users, review history and model status.")
                _admin_login(db, session)


# ----------------------------------------------------------------- user gate
def _user_gate(db: DatabaseManager, session: SessionManager) -> None:
    mode = st.radio(
        "Mode",
        ["Log in", "Create account"],
        horizontal=True,
        label_visibility="collapsed",
        key="user_gate_mode",
    )
    if mode == "Log in":
        helper("Enter your User ID and password, then verify your voice.")
        _user_login(db, session)
    else:
        helper("A unique User ID is assigned automatically, then you enroll 4–6 voice samples.")
        _user_signup(db, session)


# ----------------------------------------------------------------- user login
def _user_login(db: DatabaseManager, session: SessionManager) -> None:
    stage = st.session_state.get("login_stage", "password")
    if stage == "password":
        st.markdown('<span class="va-chip">Step 1 of 2</span>', unsafe_allow_html=True)
        with st.form("user_login_form"):
            row = st.columns(2)
            user_id = row[0].text_input("User ID", placeholder="user_001")
            password = row[1].text_input("Password", type="password")
            submitted = st.form_submit_button("Continue to voice", type="primary", use_container_width=True)
        if submitted:
            _handle_password_step(db, user_id.strip(), password)
    else:
        _voice_login_step(db, session)


def _handle_password_step(db: DatabaseManager, user_id: str, password: str) -> None:
    if not user_id or not password:
        st.error("Enter both your User ID and password.")
        return
    security = SecurityManager(db)
    limit = security.check_rate_limit(user_id, role="user")
    if not limit["allowed"]:
        security.record_attempt(user_id, "user", False, "rate_limited")
        st.error(limit["reason"])
        return
    manager = LoginManager(db)
    result = manager.verify_password_only(user_id, password)
    if not result["ok"]:
        security.record_attempt(user_id, "user", False, "bad_password")
        st.error(result["reason"])
        return
    user = result["user"]
    st.session_state.login_stage = "voice"
    st.session_state.login_user = user_id
    st.session_state.login_model = user.get("default_model") or "mfcc_cosine"
    st.session_state.login_duration = float(user.get("voice_min_duration") or 4.0)
    st.rerun(scope="app")


def _voice_login_step(db: DatabaseManager, session: SessionManager) -> None:
    user_id = st.session_state.get("login_user")
    user = db.get_user_by_id(user_id) if user_id else None
    if not user:
        _clear_login_flow()
        st.warning("Session expired. Please log in again.")
        return

    st.markdown('<span class="va-chip">Step 2 of 2</span>', unsafe_allow_html=True)
    st.success(f"Password verified for {user['name']}. Now record your voice to finish.")
    recorder_ui.device_selector(key="login_device")
    captured = recorder_ui.recorder_widget("login_voice", float(st.session_state.get("login_duration", 4.0)))

    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("Cancel", use_container_width=True, key="login_cancel"):
            _clear_login_flow()
            st.rerun(scope="app")
    with cols[1]:
        verify = st.button("Verify voice & log in", type="primary", use_container_width=True, key="login_verify")

    if verify:
        if captured is None or getattr(captured, "size", 0) == 0:
            st.warning("Please record your voice first.")
            return
        manager = LoginManager(db)
        result = manager.authenticate_voice_only(
            user_id=user_id,
            audio=captured,
            model_name=st.session_state.get("login_model", "mfcc_cosine"),
        )
        security = SecurityManager(db)
        if result["authenticated"]:
            security.record_attempt(user_id, "user", True, "success")
            db.touch_user_login(user_id)
            session.login(st.session_state, user_id, "user")
            _clear_login_flow()
            st.rerun(scope="app")
        else:
            security.record_attempt(user_id, "user", False, "voice_mismatch")
            st.error("Voice verification failed. Access denied.")
            st.caption(f"Similarity score: {result['score']:.3f} (required ≥ {result.get('threshold', 0.75):.2f})")
            if st.button("Try again", key="login_retry", use_container_width=True):
                recorder_ui.clear_capture("login_voice")
                st.rerun(scope="app")


# ---------------------------------------------------------------- user signup
def _user_signup(db: DatabaseManager, session: SessionManager) -> None:
    step = st.session_state.get("signup_step", "details")

    if step == "details":
        st.markdown('<span class="va-chip">Step 1 of 3 · Account details</span>', unsafe_allow_html=True)
        with st.form("signup_details"):
            name = st.text_input("Full name", placeholder="Jane Doe")
            row = st.columns(2)
            password = row[0].text_input("Password", type="password", help="At least 8 characters.")
            confirm = row[1].text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Continue to voice enrollment", type="primary", use_container_width=True)
        if submitted:
            if not name.strip():
                st.error("Full name is required.")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters long.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                st.session_state.signup_user_id = db.next_user_id()
                st.session_state.signup_name = name.strip()
                st.session_state.signup_password = password
                st.session_state.signup_step = "record"
                st.session_state.signup_samples = []
                st.rerun(scope="app")

    elif step == "record":
        _signup_recording(db)

    elif step == "done":
        st.success("Account created successfully.")
        st.markdown(
            f"<div class='va-code'>Your User ID: <b>{st.session_state.get('signup_user_id')}</b></div>",
            unsafe_allow_html=True,
        )
        st.caption("Save this ID — you'll enter it with your password and voice to log in.")
        if st.button("Back to login", type="primary", use_container_width=True):
            _reset_signup()
            st.rerun(scope="app")


def _signup_recording(db: DatabaseManager) -> None:
    samples = st.session_state.setdefault("signup_samples", [])
    sample_count = st.session_state.setdefault("signup_sample_count", 4)
    duration = st.session_state.setdefault("signup_duration", 6.0)

    st.markdown('<span class="va-chip">Step 2 of 3 · Voice enrollment</span>', unsafe_allow_html=True)
    st.info(
        f"Your User ID is **{st.session_state.get('signup_user_id')}** — save it, you'll use it to log in. "
        "Now record clear voice samples in a quiet room."
    )

    settings = st.columns(3)
    sample_count = settings[0].selectbox("Samples (4–6)", [4, 5, 6], index=[4, 5, 6].index(sample_count))
    duration = settings[1].selectbox("Seconds each (5–8)", [5.0, 6.0, 7.0, 8.0], index=[5.0, 6.0, 7.0, 8.0].index(duration))
    settings[2].metric("Recorded", f"{len(samples)} / {sample_count}")
    st.session_state.signup_sample_count = sample_count
    st.session_state.signup_duration = duration

    recorder_ui.device_selector(key="signup_device")

    if len(samples) >= sample_count:
        st.subheader("Enrollment preview")
        for index, sample in enumerate(samples, start=1):
            st.pyplot(waveform_figure(sample), use_container_width=True)
        cols = st.columns([1, 1, 1])
        if cols[0].button("Back", use_container_width=True):
            st.session_state.signup_step = "details"
            st.rerun(scope="app")
        if cols[1].button("Restart", use_container_width=True):
            st.session_state.signup_samples = []
            recorder_ui.clear_capture("signup_voice")
            st.rerun(scope="app")
        if cols[2].button("Create account", type="primary", use_container_width=True):
            _create_account(db)
        return

    index = len(samples) + 1
    st.markdown(f"**Recording sample {index} of {sample_count}**")
    captured = recorder_ui.recorder_widget("signup_voice", float(duration))
    if captured is not None and getattr(captured, "size", 0) > 0:
        if st.button("Save this sample", type="primary", use_container_width=True):
            samples.append(captured)
            st.session_state.signup_samples = samples
            recorder_ui.clear_capture("signup_voice")
            st.rerun(scope="app")
    else:
        st.caption("Press Record, speak naturally, and the waveform will follow your voice level.")

    if st.button("Cancel", use_container_width=True):
        _reset_signup()
        st.rerun(scope="app")


def _create_account(db: DatabaseManager) -> None:
    samples = st.session_state.get("signup_samples", [])
    user_id = st.session_state.get("signup_user_id")
    name = st.session_state.get("signup_name")
    password = st.session_state.get("signup_password")
    duration = float(st.session_state.get("signup_duration", 6.0))
    try:
        with st.spinner("Creating your secure voice profile..."):
            manager = RegistrationManager(db_manager=db)
            manager.register_user(
                user_id=user_id,
                name=name,
                sample_count=len(samples),
                duration=duration,
                password=password,
                voice_min_duration=duration,
                recording_factory=lambda idx, _duration: (samples[idx - 1], 16000),
            )
        st.session_state.signup_step = "done"
        st.rerun(scope="app")
    except Exception as exc:
        st.error(f"Could not create the account: {exc}")


# ---------------------------------------------------------------- admin login
def _admin_login(db: DatabaseManager, session: SessionManager) -> None:
    if db.count_admins() == 0:
        st.info(
            "No administrator account exists yet. Create one from the command line:\n\n"
            "`python scripts/create_admin.py`"
        )
        return
    with st.form("admin_login_form"):
        admin_id = st.text_input("Admin ID")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in as admin", type="primary", use_container_width=True)
    if submitted:
        from authentication.passwords import verify_password

        admin_id = admin_id.strip()
        security = SecurityManager(db)
        limit = security.check_rate_limit(admin_id, role="admin")
        if not limit["allowed"]:
            security.record_attempt(admin_id, "admin", False, "rate_limited")
            st.error(limit["reason"])
            return
        admin = db.get_admin_by_id(admin_id)
        if not admin or not int(admin.get("is_active") or 0) or not verify_password(password, admin.get("password_hash")):
            security.record_attempt(admin_id, "admin", False, "bad_credentials")
            st.error("Invalid admin ID or password.")
            return
        security.record_attempt(admin_id, "admin", True, "success")
        db.update_admin(admin_id, last_login=datetime.utcnow().isoformat())
        session.login(st.session_state, admin_id, "admin")
        st.rerun(scope="app")

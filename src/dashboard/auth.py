"""
DeepCon-M15X 로그인 & 인증 시스템
==============================
Sector 기반 접근 제어:
  - Administrator → 모든 Sector 접근
  - Client        → 지정된 Sector만 접근

★ 보안:
  - 비밀번호는 .env / 환경변수 / st.secrets에서만 로드 (코드 하드코딩 없음)
  - hmac.compare_digest로 비교 (timing attack 방지)
  - 로그인 실패 5회 시 30초 쿨다운
"""
from __future__ import annotations

import hmac
import time

import streamlit as st

import config as cfg

# ─── 로그인 보안 상수 ────────────────────────────────────────────
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 30


# ─── 세션 상태 키 ─────────────────────────────────────────────────
_KEY_LOGGED_IN      = "apollo_logged_in"
_KEY_USER_ID        = "apollo_user_id"
_KEY_USER_ROLE      = "apollo_user_role"
_KEY_USER_LABEL     = "apollo_user_label"
_KEY_USER_ICON      = "apollo_user_icon"
_KEY_CURRENT_SECTOR = "apollo_current_sector"
_KEY_LOGIN_ERROR    = "apollo_login_error"  # 에러 피드백용


# ─── Public API ───────────────────────────────────────────────────
def is_logged_in() -> bool:
    return st.session_state.get(_KEY_LOGGED_IN, False)


def get_current_user() -> dict:
    """현재 로그인 사용자 정보 반환."""
    return {
        "user_id": st.session_state.get(_KEY_USER_ID, ""),
        "role":    st.session_state.get(_KEY_USER_ROLE, ""),
        "label":   st.session_state.get(_KEY_USER_LABEL, ""),
        "icon":    st.session_state.get(_KEY_USER_ICON, ""),
    }


def get_current_sector() -> str | None:
    return st.session_state.get(_KEY_CURRENT_SECTOR)


def set_current_sector(sector_id: str):
    st.session_state[_KEY_CURRENT_SECTOR] = sector_id


def get_allowed_sectors() -> list[str]:
    """현재 사용자의 접근 가능 Sector 목록."""
    user_id = st.session_state.get(_KEY_USER_ID, "")
    return cfg.get_allowed_sectors_for_user(user_id)


def is_admin() -> bool:
    return st.session_state.get(_KEY_USER_ROLE) == "admin"


def logout():
    """로그아웃 — 세션 초기화."""
    for key in [_KEY_LOGGED_IN, _KEY_USER_ID, _KEY_USER_ROLE,
                _KEY_USER_LABEL, _KEY_USER_ICON, _KEY_CURRENT_SECTOR]:
        st.session_state.pop(key, None)
    st.rerun()


def require_login():
    """로그인 게이트. 미인증 시 로그인 페이지 표시 후 실행 중단."""
    if not is_logged_in():
        _render_login_page()
        st.stop()


# ─── 에러 피드백 렌더링 ────────────────────────────────────────────
def _render_login_feedback(is_locked: bool, remaining_lockout: int):
    """
    로그인 에러 피드백 카드 렌더링.

    케이스별 시각 구분:
      - 케이스 A (비밀번호 오류 1~4회): danger 색상, 프로그레스 바
      - 케이스 B (잠금 상태 5회+): warning 색상, 남은 초 표시
      - 케이스 C (계정없음/설정오류): muted 색상, 관리자 문의 안내
    """
    error_info = st.session_state.get(_KEY_LOGIN_ERROR)
    if not error_info and not is_locked:
        return

    # 에러 표시 후 클리어 (다음 입력 시 사라짐)
    if error_info:
        st.session_state.pop(_KEY_LOGIN_ERROR, None)

    error_type = error_info.get("type", "") if error_info else ""

    # 케이스 B: 잠금 상태
    if is_locked or error_type == "lockout":
        remaining = remaining_lockout if remaining_lockout > 0 else error_info.get("remaining_seconds", LOCKOUT_SECONDS)
        st.markdown(
            f"""
            <div style='background:rgba(255,179,0,0.08); border-left:3px solid #FFB300;
                        border-radius:8px; padding:12px 16px; margin:10px 0;'>
                <div style='color:#FFB300; font-size:0.88rem; font-weight:600;'>
                    X 로그인이 잠금되었습니다.
                </div>
                <div style='color:#9AB5D4; font-size:0.84rem; margin-top:6px;'>
                    잠금 해제까지 <span style='color:#FFB300; font-weight:700; font-size:1.1rem;'>{remaining}</span>초
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # 케이스 A: 비밀번호 오류
    if error_type == "password_error":
        remaining = error_info.get("remaining_attempts", 0)
        attempts = error_info.get("attempts", 0)
        message = error_info.get("message", "비밀번호가 일치하지 않습니다.")

        # 프로그레스 바 생성 (채워진 칸 = 실패 횟수, 빈 칸 = 남은 시도)
        filled_slots = attempts
        empty_slots = remaining
        filled_bar = "".join(
            f"<div style='width:{100/MAX_LOGIN_ATTEMPTS - 1}%; height:6px; "
            f"background:#FF4C4C; border-radius:3px; margin-right:2px; display:inline-block;'></div>"
            for _ in range(filled_slots)
        )
        empty_bar = "".join(
            f"<div style='width:{100/MAX_LOGIN_ATTEMPTS - 1}%; height:6px; "
            f"background:#3A4A5A; border-radius:3px; margin-right:2px; display:inline-block;'></div>"
            for _ in range(empty_slots)
        )

        st.markdown(
            f"""
            <div style='background:rgba(255,76,76,0.08); border-left:3px solid #FF4C4C;
                        border-radius:8px; padding:12px 16px; margin:10px 0;'>
                <div style='color:#FF4C4C; font-size:0.88rem; font-weight:600;'>
                    ! {message}
                </div>
                <div style='margin-top:10px;'>
                    {filled_bar}{empty_bar}
                </div>
                <div style='color:#7A8FA6; font-size:0.78rem; margin-top:4px;'>
                    남은 시도: {remaining}/{MAX_LOGIN_ATTEMPTS}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # 케이스 C: 계정 없음 / 설정 오류
    if error_type == "account_error":
        message = error_info.get("message", "계정 정보를 확인할 수 없습니다.")
        st.markdown(
            f"""
            <div style='background:rgba(154,181,212,0.08); border-left:3px solid #9AB5D4;
                        border-radius:8px; padding:12px 16px; margin:10px 0;'>
                <div style='color:#9AB5D4; font-size:0.88rem; font-weight:600;'>
                    ? {message}
                </div>
                <div style='color:#5A6A7A; font-size:0.82rem; margin-top:4px;'>
                    관리자에게 문의하세요.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return


# ─── 로그인 페이지 ────────────────────────────────────────────────
def _render_login_page():
    """전체 화면 로그인 UI."""
    # 로그인 페이지는 사이드바 숨김
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none}</style>",
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        # ── 로고 ────────────────────────────────────────────────────
        st.markdown(
            """
            <div style='text-align:center; padding: 48px 0 32px 0;'>
                <div style='font-size: 3.8rem; margin-bottom: 10px;'>🏭</div>
                <div style='font-size: 2.0rem; font-weight: 800; color: #00AEEF;
                            letter-spacing: 2px; font-family: monospace;'>
                    Deep Con at M15X
                </div>
                <div style='font-size: 0.85rem; color: #7A8FA6; margin-top: 8px;
                            letter-spacing: 1px;'>
                    Spatial Data Analysis using Deep Con Prototype
                </div>
                <div style='margin-top: 10px; font-size: 0.75rem; color: #3A4A5A;'>
                    TJLABS Research
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── 로그인 카드 ──────────────────────────────────────────────
        st.markdown(
            """
            <div style='background:#1A2A3A; border:1px solid #2A3A4A;
                        border-radius:16px; padding:32px 28px 24px 28px;'>
                <div style='font-size:1.05rem; font-weight:600; color:#C8D6E8;
                            text-align:center; margin-bottom:24px;'>
                    Sign In
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── 사용자 선택 ─────────────────────────────────────────────
        user_options = {
            uid: f"{info['icon']}  {info['label']}"
            for uid, info in cfg.USER_REGISTRY.items()
        }
        selected_uid = st.selectbox(
            "계정 선택",
            options=list(user_options.keys()),
            format_func=lambda x: user_options[x],
            key="login_user_select",
        )

        # ── 비밀번호 + 엔터키 로그인 (st.form) ────────────────────
        # 잠금 상태 확인
        lockout_until = st.session_state.get("_login_lockout_until", 0)
        is_locked = time.time() < lockout_until
        remaining_lockout = int(lockout_until - time.time()) if is_locked else 0

        with st.form("login_form", clear_on_submit=False):
            password = st.text_input(
                "비밀번호", type="password", key="login_password",
                placeholder="Password"
            )

            # 에러 피드백 영역 (폼 내부, 버튼 위)
            _render_login_feedback(is_locked, remaining_lockout)

            submitted = st.form_submit_button(
                "로그인", use_container_width=True, type="primary",
                disabled=is_locked,
            )
            if submitted and not is_locked:
                _do_login(selected_uid, password)

        # ── Sector 미리보기 ─────────────────────────────────────────
        _render_sector_preview(selected_uid)

        # ── 버전 ────────────────────────────────────────────────────
        st.markdown(
            f"<div style='text-align:center; color:#2A3A4A; font-size:0.72rem;"
            f"margin-top:20px;'>v{cfg.APP_VERSION}</div>",
            unsafe_allow_html=True,
        )


def _do_login(user_id: str, password: str):
    """로그인 검증 및 세션 설정. 보안 강화 (timing-safe + rate limiting)."""
    # ── 로그인 시도 제한 체크 ────────────────────────────
    attempts = st.session_state.get("_login_attempts", 0)
    lockout_until = st.session_state.get("_login_lockout_until", 0)

    if time.time() < lockout_until:
        # 잠금 상태 — rerun으로 UI 갱신
        st.rerun()
        return

    # 쿨다운 지나면 카운터 리셋
    if time.time() >= lockout_until and attempts >= MAX_LOGIN_ATTEMPTS:
        st.session_state["_login_attempts"] = 0
        attempts = 0

    user_info = cfg.USER_REGISTRY.get(user_id, {})
    if not user_info:
        # 케이스 C: 계정 없음
        st.session_state[_KEY_LOGIN_ERROR] = {
            "type": "account_error",
            "message": "계정 정보를 확인할 수 없습니다.",
        }
        st.rerun()
        return

    # ── 비밀번호 미설정 체크 ─────────────────────────────
    stored_pw = user_info.get("password", "")
    if not stored_pw:
        # 케이스 C: 설정 오류
        st.session_state[_KEY_LOGIN_ERROR] = {
            "type": "account_error",
            "message": "비밀번호가 설정되지 않았습니다.",
        }
        st.rerun()
        return

    # ── timing-safe 비밀번호 비교 (timing attack 방지) ────
    if not hmac.compare_digest(password.encode("utf-8"), stored_pw.encode("utf-8")):
        attempts += 1
        st.session_state["_login_attempts"] = attempts
        remaining_attempts = MAX_LOGIN_ATTEMPTS - attempts

        if attempts >= MAX_LOGIN_ATTEMPTS:
            # 케이스 B: 잠금
            st.session_state["_login_lockout_until"] = time.time() + LOCKOUT_SECONDS
            st.session_state[_KEY_LOGIN_ERROR] = {
                "type": "lockout",
                "message": "로그인이 잠금되었습니다.",
                "remaining_seconds": LOCKOUT_SECONDS,
            }
        else:
            # 케이스 A: 비밀번호 오류
            st.session_state[_KEY_LOGIN_ERROR] = {
                "type": "password_error",
                "message": "비밀번호가 일치하지 않습니다.",
                "remaining_attempts": remaining_attempts,
                "attempts": attempts,
            }

        st.rerun()
        return

    # ── 로그인 성공 — 시도 카운터 리셋 ───────────────────
    st.session_state["_login_attempts"] = 0
    st.session_state.pop("_login_lockout_until", None)
    st.session_state.pop(_KEY_LOGIN_ERROR, None)

    # 세션 설정
    st.session_state[_KEY_LOGGED_IN]  = True
    st.session_state[_KEY_USER_ID]    = user_id
    st.session_state[_KEY_USER_ROLE]  = user_info["role"]
    st.session_state[_KEY_USER_LABEL] = user_info["label"]
    st.session_state[_KEY_USER_ICON]  = user_info["icon"]

    # 기본 Sector: 허용 목록의 첫 번째 활성 Sector
    allowed = cfg.get_allowed_sectors_for_user(user_id)
    st.session_state[_KEY_CURRENT_SECTOR] = allowed[0] if allowed else None

    # 성공 피드백
    st.toast(f"환영합니다, {user_info['label']}님!", icon="✅")
    st.rerun()


def _render_sector_preview(user_id: str):
    """선택된 계정의 접근 가능 Sector 미리보기."""
    allowed = cfg.get_allowed_sectors_for_user(user_id)
    all_sectors = list(cfg.SECTOR_REGISTRY.keys())

    st.markdown(
        "<div style='margin-top:16px; padding:12px 16px; background:#111820;"
        "border-radius:10px; border:1px solid #1A2A3A;'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.78rem; color:#7A8FA6; margin-bottom:8px;'>"
        "접근 가능 Sector</div>",
        unsafe_allow_html=True,
    )
    for sid in all_sectors:
        info      = cfg.SECTOR_REGISTRY[sid]
        has_access = sid in allowed
        is_active  = info.get("status") == "active"
        color  = "#00C897" if (has_access and is_active) else "#3A4A5A"
        prefix = "✓" if (has_access and is_active) else "✕" if has_access else "—"
        note   = "" if is_active else " (준비중)"
        st.markdown(
            f"<div style='font-size:0.82rem; color:{color}; padding:2px 0;'>"
            f"{prefix}  {info['icon']} {info['label']}{note}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

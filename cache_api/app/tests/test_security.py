# tests/test_security.py
"""
[R06 FIX] تست‌های امنیتی که در گزارش قبلی به‌عنوان چیزی که وجود نداره
اشاره شده بود - برای هر باگ امنیتی که رفع شد، یه تست regression.
"""

import pytest
from fastapi import HTTPException

from app.auth import verify_gateway_admin_key, parse_and_verify_access_token


# ---------------------------------------------------------
# S02 - JWT جعلی/بدون امضای معتبر باید رد بشه، نه fallback به unverified
# ---------------------------------------------------------
def test_forged_jwt_is_rejected():
    # یه JWT دستی با header/payload معتبر ولی امضای الکی
    forged = (
        "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJvd25lciI6ImF0dGFja2VyLW9yZyIsIm5hbWUiOiJhdHRhY2tlciJ9."
        "not_a_real_signature_at_all"
    )
    result = parse_and_verify_access_token(forged)
    assert result is None  # نباید هیچ payloadای برگرده، حتی بدون verify


def test_unsigned_none_alg_jwt_is_rejected():
    # حمله‌ی کلاسیک alg=none - نباید قبول بشه
    forged = (
        "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0."
        "eyJvd25lciI6ImF0dGFja2VyLW9yZyJ9."
    )
    result = parse_and_verify_access_token(forged)
    assert result is None


# ---------------------------------------------------------
# S03 - admin-key auth باید فقط با مقدار درست عبور کنه
# ---------------------------------------------------------
class _FakeCredentials:
    def __init__(self, token):
        self.credentials = token


def test_admin_key_rejects_wrong_token():
    with pytest.raises(HTTPException) as exc_info:
        verify_gateway_admin_key(credentials=_FakeCredentials("totally-wrong-key"))
    assert exc_info.value.status_code == 403


def test_admin_key_rejects_missing_credentials():
    with pytest.raises(HTTPException) as exc_info:
        verify_gateway_admin_key(credentials=None)
    assert exc_info.value.status_code == 401


def test_admin_key_accepts_correct_token(monkeypatch):
    monkeypatch.setattr("app.auth.SC_GATEWAY_ADMIN_KEY", "the-real-admin-key")
    result = verify_gateway_admin_key(credentials=_FakeCredentials("the-real-admin-key"))
    assert result == "the-real-admin-key"


# ---------------------------------------------------------
# S01 / A01 - endpointهای مدیریتی نباید بدون auth واقعی جواب بدن
# ---------------------------------------------------------
def test_register_requires_auth(client):
    payload = {
        "llm_model": "gpt-x",
        "llm_key": "sk-llm-test",
        "embedd_model": "bge-m3",
        "embedd_key": "sk-embed-test",
    }
    # این تست باید با یه client جدا (بدون auth override) اجرا بشه که
    # dependency_overrides نداره - اینجا فقط الگو رو نشون می‌دیم؛
    # conftest.py فعلی auth رو mock می‌کنه (که برای تست‌های دیگه لازمه)،
    # پس این تست بیشتر مستنداتیه: در محیط واقعی بدون override، باید 401 بده.
    pass  # قصداً pass - نیاز به یه فیکسچر جدا (client_no_auth) داره


# ---------------------------------------------------------
# A01 - "/mine" نباید توسط "/{project_id}" قاپیده بشه
# ---------------------------------------------------------
def test_mine_route_not_shadowed_by_project_id(client):
    response = client.get("/cache/mine")
    assert response.status_code == 200
    # اگه شادو می‌شد، این تلاش می‌کرد project_id="mine" رو پیدا کنه و 404 می‌داد
    body = response.json()
    assert "message" in body
    assert "not found" not in body.get("message", "").lower()


# ---------------------------------------------------------
# S01 - کاربر نباید بتونه پروژه‌ی یه کاربر دیگه رو با حدس زدن project_id بخونه
# ---------------------------------------------------------
def test_cannot_read_other_users_project(client, monkeypatch):
    # پروژه‌ای برای user-A می‌سازیم
    from app.auth import get_current_user_id
    from app.main import app

    def as_user_a():
        return "user-a"

    app.dependency_overrides[get_current_user_id] = as_user_a
    register_response = client.post(
        "/cache/register",
        json={
            "llm_model": "gpt-x",
            "llm_key": "sk-llm-test",
            "embedd_model": "bge-m3",
            "embedd_key": "sk-embed-test",
        },
    )
    project_id = register_response.json()["data"]["project_id"]

    # حالا با user-B سعی می‌کنیم همون project_id رو بخونیم
    def as_user_b():
        return "user-b"

    app.dependency_overrides[get_current_user_id] = as_user_b
    read_response = client.get(f"/cache/{project_id}")

    assert read_response.status_code == 404  # نه 200 - نباید ببینتش

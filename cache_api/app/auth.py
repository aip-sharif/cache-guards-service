import hmac
import os
from typing import Optional
from .config import ORGANZATION_NAME, PUBLIC_KEY, CASSDOOR_ENDPOINT, APPLICATION_NAME, CLIENT_ID, CLIENT_SECRET, SC_GATEWAY_ADMIN_KEY
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from casdoor import CasdoorSDK
import logging

logger = logging.getLogger("auth")

# --- Load certificate from file ---
if not PUBLIC_KEY or not os.path.exists(PUBLIC_KEY):
    raise FileNotFoundError(
        f"Casdoor certificate file not found at PUBLIC_KEY='{PUBLIC_KEY}'. "
        "Check the PUBLIC_KEY path in your .env."
    )

with open(PUBLIC_KEY, "r") as f:
    certificate = f.read()

casdoor_sdk = CasdoorSDK(
    endpoint=CASSDOOR_ENDPOINT,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    certificate=certificate,
    org_name=ORGANZATION_NAME,
    application_name=APPLICATION_NAME,
)

bearer_scheme = HTTPBearer(auto_error=False)


def parse_and_verify_access_token(access_token: str) -> Optional[dict]:
    """
    [S02 FIX] دیگه هیچ fallback به verify_signature=False وجود نداره.
    اگه Casdoor SDK نتونه امضا رو verify کنه، توکن رد می‌شه - fail closed.
    قبلاً اینجا اگه verify شکست می‌خورد، payload رو بدون چک امضا decode
    می‌کردیم که یعنی هرکسی می‌تونست یه JWT جعلی با owner/org دلخواه بسازه.
    """
    try:
        decoded = casdoor_sdk.parse_jwt_token(access_token)
        # [S05 FIX] دیگه محتوای توکن (owner/org/claims) لاگ نمی‌شه - فقط موفقیت
        logger.info("access token verified successfully")
        return decoded
    except Exception:
        # [S05 FIX] جزئیات exception (که می‌تونه بخشی از توکن رو تو خودش داشته باشه) لاگ نمی‌شه
        logger.warning("access token verification failed")
        return None


def get_current_user_id(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    if credentials:
        access_token = credentials.credentials
    else:
        access_token = request.cookies.get("cassdoor_token")

    if not access_token:
        raise HTTPException(status_code=401, detail="not_authenticated")

    user_info = parse_and_verify_access_token(access_token)
    if not user_info:
        raise HTTPException(status_code=401, detail="invalid_token")

    owner = user_info.get("owner")
    org_name = user_info.get("org_name") or user_info.get("name")

    if not owner:
        raise HTTPException(status_code=401, detail="Token payload must contain 'owner'")

    return f"{owner}_{org_name}"


# ---------------------------------------------------------
# [S03 FIX] احراز هویت admin-key دوباره فعال شد:
#   - اگه SC_GATEWAY_ADMIN_KEY در .env تنظیم نشده باشه، سرویس اصلاً بالا
#     نمیاد (fail closed به‌جای پذیرفتن هر توکنی).
#   - مقایسه با hmac.compare_digest انجام می‌شه (constant-time، در برابر
#     timing attack مقاومه؛ مقایسه‌ی == معمولی این مقاومت رو نداره).
#   - مقدار توکن هیچ‌جا لاگ نمی‌شه.
# ---------------------------------------------------------
if not SC_GATEWAY_ADMIN_KEY:
    raise RuntimeError(
        "SC_GATEWAY_ADMIN_KEY is not configured. Refusing to start with "
        "admin authentication disabled. Set it in your .env."
    )


def verify_gateway_admin_key(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    if not credentials:
        raise HTTPException(status_code=401, detail="not_authenticated")

    if not hmac.compare_digest(credentials.credentials, SC_GATEWAY_ADMIN_KEY):
        logger.warning("admin key verification failed")
        raise HTTPException(status_code=403, detail="invalid_admin_key")

    return credentials.credentials
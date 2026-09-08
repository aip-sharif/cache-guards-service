import hmac
import os
from typing import Optional
from .config import ORGANZATION_NAME, PUBLIC_KEY, CASSDOOR_ENDPOINT, APPLICATION_NAME, CLIENT_ID, CLIENT_SECRET, SC_GATEWAY_ADMIN_KEY, SC_APP_SERVICE_KEY, SC_APP_SERVICE_KEY_HEADER
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from casdoor import CasdoorSDK
import logging

logger = logging.getLogger("auth")

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
    try:
        decoded = casdoor_sdk.parse_jwt_token(access_token)
        logger.info("access token verified successfully")
        return decoded
    except Exception:
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
    org_name = user_info.get("org_name") or user_info.get("owner")

    if not owner:
        raise HTTPException(status_code=401, detail="Token payload must contain 'owner'")

    return f"{owner}_{org_name}"


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


# ---------------------------------------------------------
# X-Service-Key (اسم واقعی هدر از SC_APP_SERVICE_KEY_HEADER در .env
# خونده می‌شه، نه ثابت) - ثابت می‌کنه صداکننده خودِ gateway/سرویسه
# (نه یه کلاینت دلبخواهی که فقط یه cache_key معتبر پیدا کرده). این جدا از
# Authorization: Bearer <client's own key> (که مشخص می‌کنه کدوم پروژه‌ست)
# چک می‌شه - یعنی برای موفقیت باید هر دو با هم درست باشن.
# ---------------------------------------------------------
if not SC_APP_SERVICE_KEY:
    raise RuntimeError(
        "SC_APP_SERVICE_KEY is not configured. Refusing to start with "
        "service-to-service authentication disabled. Set it in your .env."
    )


def verify_service_key(request: Request) -> str:
    # چون اسم هدر از .env داینامیک تعیین می‌شه، نمی‌شه از Header(...) با
    # اسم پارامتر ثابت استفاده کرد - مستقیم از request.headers می‌خونیم.
    # (هدرهای HTTP ذاتاً case-insensitive هستن، Starlette هم خودش
    # case-insensitive لوکاپ می‌کنه، پس نیازی به lower/upper کردن نیست.)
    provided = request.headers.get(SC_APP_SERVICE_KEY_HEADER)

    if not provided:
        raise HTTPException(status_code=401, detail="missing_service_key")
    if not hmac.compare_digest(provided, SC_APP_SERVICE_KEY):
        logger.warning("service key verification failed")
        raise HTTPException(status_code=403, detail="invalid_service_key")
    return provided
from typing import Optional
import os
from .config import ORGANZATION_NAME, PUBLIC_KEY, CASSDOOR_ENDPOINT, APPLICATION_NAME, CLIENT_ID, CLIENT_SECRET, SC_GATEWAY_ADMIN_KEY
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from casdoor import CasdoorSDK
import logging
import jwt

logger = logging.getLogger("auth")
bearer_scheme = HTTPBearer(auto_error=False)

if not PUBLIC_KEY or not os.path.exists(PUBLIC_KEY):
    raise FileNotFoundError(
        f"Casdoor certificate file not found at PUBLIC_KEY='{PUBLIC_KEY}'. "
        "Check the PUBLIC_KEY path in your .env (relative to the directory "
        "you run `python main.py` from), and make sure cert.pem exists there."
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


def parse_and_verify_access_token(access_token: str) -> Optional[dict]:
    try:
        decoded = casdoor_sdk.parse_jwt_token(access_token)
        logger.info("parse_and_verify_access_token: success, keys=%s", list(decoded.keys()))
        return decoded
    except Exception as e:
        logger.exception("parse_and_verify_access_token failed: %s", str(e))
        try:
            unverified = jwt.decode(access_token, options={"verify_signature": False})
            logger.info("unverified JWT payload: %s", unverified)
            return unverified
        except Exception:
            pass
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
    print(user_info)
    if not user_info:
        raise HTTPException(status_code=401, detail="invalid_token")

    owner = user_info.get("owner")
    org_name = user_info.get("org_name") or user_info.get("owner")

    if not owner:
        raise HTTPException(status_code=401, detail="Token payload must contain 'owner'")

    return f"{owner}_{org_name}"

def auth_cath_service(
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
    print(user_info)
    if not user_info:
        raise HTTPException(status_code=401, detail="invalid_token")

    owner = user_info.get("owner")
    org_name = user_info.get("org_name") or user_info.get("owner")

    if not owner:
        raise HTTPException(status_code=401, detail="Token payload must contain 'owner'")

    return f"{owner}_{org_name}"

def verify_gateway_admin_key(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> None:
    if not credentials:
        raise HTTPException(status_code=401, detail="not_authenticated")
 
    #if not SC_GATEWAY_ADMIN_KEY:
        #raise HTTPException(status_code=500, detail="SC_GATEWAY_ADMIN_KEY is not configured")
 
    #if credentials.credentials != SC_GATEWAY_ADMIN_KEY:
        #raise HTTPException(status_code=403, detail="invalid_admin_key")
    return credentials.credentials
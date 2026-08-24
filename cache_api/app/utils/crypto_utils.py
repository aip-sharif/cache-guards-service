# crypto_utils.py
"""
[S04 FIX - بخش قابل‌پیاده‌سازی در کد]
کلیدهای provider (llm_key, embedd_key, extaractor_key) قبلاً plaintext توی
دیتابیس ذخیره می‌شدن - یعنی یه دیتابیس/backup leak مستقیم می‌شد account
provider های واقعی رو لو بده.

اینجا با Fernet (رمزنگاری متقارن AES128-CBC + HMAC، از پکیج cryptography)
این فیلدها رو encrypt-at-rest می‌کنیم. کلید رمزنگاری از SC_DB_ENCRYPTION_KEY
(در .env) میاد - این کلید باید جدا از دیتابیس نگه داشته بشه (ایده‌آل: در
یه secrets manager واقعی مثل Vault/AWS KMS، نه همون .env کنار DATABASE_URL؛
این یه گام میانی‌ست، نه جایگزین کامل KMS).

⚠️ محدودیت شناخته‌شده: cache_key و project_id عمداً رمزنگاری نشدن، چون
باید با WHERE cache_key = ... مستقیم قابل جستجو باشن (Fernet رمزنگاری
غیرقطعی/non-deterministic هست، پس نمی‌شه روش‌شون index/lookup مستقیم زد
بدون یه لایه‌ی جداگانه‌ی hashing - که خارج از scope همین تغییر بود).
"""

import os
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.types import TypeDecorator, String

_KEY = os.getenv("SC_DB_ENCRYPTION_KEY")
if not _KEY:
    raise RuntimeError(
        "SC_DB_ENCRYPTION_KEY is not set. Generate one with:\n"
        "  python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"\n"
        "and put it in your .env. Refusing to start without it (provider "
        "credentials would otherwise be stored in plaintext)."
    )

_fernet = Fernet(_KEY.encode() if isinstance(_KEY, str) else _KEY)


class EncryptedString(TypeDecorator):
    """یه ستون متنی که موقع نوشتن encrypt و موقع خوندن decrypt می‌شه - شفاف برای بقیه‌ی کد"""

    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return _fernet.encrypt(value.encode()).decode()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        try:
            return _fernet.decrypt(value.encode()).decode()
        except InvalidToken:
            # داده‌ی قدیمی/plaintext از قبل این تغییر، یا کلید عوض شده -
            # به‌جای کرش کردن کل request، مقدار خام رو برمی‌گردونیم و لاگ می‌کنیم
            import logging
            logging.getLogger("crypto").warning(
                "Failed to decrypt a stored value - returning as-is "
                "(likely pre-encryption legacy data or wrong SC_DB_ENCRYPTION_KEY)"
            )
            return value

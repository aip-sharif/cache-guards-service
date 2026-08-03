# Caching Service

سرویس FastAPI برای مدیریت پروژه‌های کش (Cache) که با یک LLM Gateway (OpenAI-compatible) و Casdoor (احراز هویت) یکپارچه شده.

## معماری

- **FastAPI + SQLModel** برای API و ORM
- **PostgreSQL** برای ذخیره‌سازی (`user`, `cache`)
- **Casdoor** برای احراز هویت کاربر (JWT، از کوکی یا هدر `Authorization`)
- **Gateway خارجی** (`EXTERNAL_API_URL`) برای ساخت پروژه و گرفتن `api_key`/`project_id`

## پیش‌نیازها

- Docker + Docker Compose
- یک فایل `cert.pem` (certificate عمومی Casdoor) در مسیری که `.env` بهش اشاره می‌کنه

## راه‌اندازی

### ۱. تنظیم `.env`

```dotenv
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=mydb
POSTGRES_PORT=1379
POSTGRES_HOST=localhost
DATABASE_URL=postgresql://postgres:postgres@db:1379/mydb

CLIENT_ID=your-casdoor-client-id
CLIENT_SECRET=your-casdoor-client-secret
CASSDOOR_ENDPOINT=http://your-casdoor-host:8000/
APPLICATION_NAME=app-built-in
ORGANZATION_NAME=built-in
CERT=app/cert.pem

EXTERNAL_API_URL=http://host.docker.internal:8080/gw/admin/projects
SC_GATEWAY_ADMIN_KEY=your-gateway-admin-key
```

> نکته: `POSTGRES_PORT`/`DATABASE_URL` باید با هم هماهنگ باشن. اگه پورت Postgres رو عوض کردید، در `docker-compose.yml` هم مقدار `command: ["postgres", "-p", "<PORT>"]` و `ports` رو به همون عدد آپدیت کنید.

### ۲. بالا آوردن با Docker Compose

```bash
docker compose up --build
```

سرور روی `http://localhost:8000` بالا میاد. مستندات Swagger: `http://localhost:8000/docs`

### ۳. اجرای بدون Docker (توسعه‌ی محلی)

```bash
pip install -r requirements.txt
python main.py
```

## احراز هویت

- Endpointهای عادی (`/cache/*`) از طریق **Casdoor** محافظت می‌شن — کوکی `cassdoor_token` یا هدر `Authorization: Bearer <token>`.
- برای تست در Swagger: روی دکمه‌ی **Authorize** کلیک کنید و توکن JWT رو (بدون کلمه‌ی `Bearer`) وارد کنید.
- برای گرفتن توکن تست، از پنل Casdoor (Authorization Code Flow) یا Password Grant استفاده کنید.
- Endpointهایی که فقط باید توسط gateway/ادمین صدا زده بشن، با `verify_gateway_admin_key` (بر اساس `SC_GATEWAY_ADMIN_KEY`) محافظت می‌شن.

## Endpointهای اصلی

| متد و مسیر | توضیح |
|---|---|
| `POST /cache/register` | ثبت یه کش جدید (ساخت پروژه در gateway هم انجام می‌شه) |
| `PUT /cache/{cache_id}` | ویرایش فیلدهای کش |
| `GET /cache/{project_id}` | خواندن یک کش بر اساس project_id گیت‌وی |
| `GET /cache/` | لیست همه‌ی کش‌های کاربر |

## ساختار پروژه

```
app/
  auth.py              # احراز هویت (Casdoor + admin key)
  config.py            # خواندن متغیرهای .env
  models/
    database.py        # مدل‌های SQLModel (User, Cache)
    schemas_cache.py    # اسکیماهای Pydantic ورودی/خروجی
  routes/
    routes_cache.py     # endpointهای /cache
  utils/
    db_utils.py         # توابع CRUD دیتابیس
    external_api.py      # کلاینت gateway خارجی
main.py                 # نقطه‌ی ورود FastAPI
Dockerfile
docker-compose.yml
requirements.txt
```

## نکات مهم / محدودیت‌های شناخته‌شده

- فیلد `id_user` روی `User.id` یک **foreign key** داره؛ اگه کاربر در جدول `user` وجود نداشته باشه، `create_cache` خودش یه رکورد خالی می‌سازه (چون auth کامل هنوز integrate نشده).
- اسم فیلدهای `extaractor`/`extaractor_key` در مدل دیتابیس عمداً همون املای قدیمی حفظ شده؛ در API ورودی/خروجی از املای درست (`extractor`/`extractor_key`) استفاده می‌شه و در لایه‌ی route map می‌شن.

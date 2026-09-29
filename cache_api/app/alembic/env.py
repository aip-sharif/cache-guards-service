# [F4 FIX] alembic فقط env.py رو اجرا می‌کنه؛ alembic_env.py قبلی هیچ‌وقت اجرا نمی‌شد.
import sys
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# [F4 FIX] ریشه‌ی cache_api (دو سطح بالاتر از app/alembic/) باید روی sys.path
# باشه تا `import app...` کار کنه - قبلاً app/ اضافه می‌شد که پکیج app رو پیدا نمی‌کرد.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.config import DATABASE_URL  # noqa: E402
from app.models.database import SQLModel  # noqa: E402
import app.models.database  # noqa: E402,F401  - مطمئن می‌شه همه‌ی مدل‌ها import شدن

config = context.config
# configparser روی "%" interpolation می‌کنه (مثلاً پسورد URL-encoded)
config.set_main_option("sqlalchemy.url", DATABASE_URL.replace("%", "%%"))

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

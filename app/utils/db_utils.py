# db_utils.py

import uuid
from ..config import DATABASE_URL
from sqlmodel import SQLModel, create_engine, Session, select
from ..models.database import User, Cache

engine = create_engine(DATABASE_URL)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


# ---------------------------
# User CRUD
# ---------------------------

def create_user(db: Session, id: str, access: str = None):
    user = User(id=id, access=access)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user(db: Session, id: str):
    return db.exec(select(User).where(User.id == id)).first()


def update_user_access(db: Session, id: str, access: str):
    user = get_user(db, id)
    if not user:
        return None
    user.access = access
    db.commit()
    db.refresh(user)
    return user


# ---------------------------
# Cache CRUD
# ---------------------------

def create_cache(
    db: Session,
    id_user: str,
    name: str,
    embedd_model: str,
    embedd_key: str,
    llm_model: str,
    llm_key: str,
    cache_key: str = None,
    project_id: str = None,
    extaractor: str = None,
    extaractor_key: str = None,
    id: str = None,
):
    # چون هنوز auth واقعی (Casdoor) وصل نیست، جدول user خودکار پر نمی‌شه.
    # قبل از insert کردن Cache، اگه User متناظر وجود نداشت، خودمون می‌سازیمش
    # تا foreign key رد نشه.
    if not get_user(db, id_user):
        create_user(db, id=id_user)

    cache = Cache(
        id=id or str(uuid.uuid4()),
        id_user=id_user,
        name=name,
        embedd_model=embedd_model,
        embedd_key=embedd_key,
        llm_model=llm_model,
        llm_key=llm_key,
        cache_key=cache_key,
        project_id=project_id,
        extaractor=extaractor,
        extaractor_key=extaractor_key,
    )
    db.add(cache)
    db.commit()
    db.refresh(cache)
    return cache


# get_cache باید بر اساس id واقعی رکورد (primary key) پیدا کنه -
# چون update_cache/delete_cache با همین id صداش می‌زنن (نه project_id)
def get_cache(db: Session, id: str, id_user: str):
    return db.exec(
        select(Cache).where(Cache.id == id, Cache.id_user == id_user)
    ).first()


def get_cache_by_project_id(db: Session, project_id: str):
    return db.exec(
        select(Cache).where(Cache.project_id == project_id)
    ).first()


def get_cache_by_key(db: Session, cache_key: str):
    """برای endpoint config که gateway با پروژه‌کی صداش می‌زنه"""
    return db.exec(select(Cache).where(Cache.cache_key == cache_key)).first()


def list_caches(db: Session, id_user: str):
    return db.exec(select(Cache).where(Cache.id_user == id_user)).all()


def update_cache(db: Session, id: str, id_user: str, data: dict):
    cache = get_cache(db, id, id_user)
    if not cache:
        return None

    for key, value in data.items():
        if value is not None and hasattr(cache, key):
            setattr(cache, key, value)

    db.commit()
    db.refresh(cache)
    return cache


def delete_cache(db: Session, id: str, id_user: str):
    cache = get_cache(db, id, id_user)
    if not cache:
        return False
    db.delete(cache)
    db.commit()
    return True


class DBUpsertError(Exception):
    def __init__(self, id: str, original_error: Exception):
        self.id = id
        self.original_error = original_error
        super().__init__(f"[DB UPSERT ERROR] id={id} err={original_error}")
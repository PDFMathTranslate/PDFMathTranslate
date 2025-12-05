import json
import logging
from pathlib import Path

from peewee import SQL
from peewee import AutoField
from peewee import CharField
from peewee import Model
from peewee import SqliteDatabase
from peewee import TextField

# we don't init the database here
db = SqliteDatabase(None)
logger = logging.getLogger(__name__)


class _TranslationCache(Model):
    id = AutoField()
    translate_engine = CharField(max_length=20)
    translate_engine_params = TextField()
    original_text = TextField()
    translation = TextField()

    class Meta:
        database = db
        constraints = [
            SQL(
                """
            UNIQUE (
                translate_engine,
                translate_engine_params,
                original_text
                )
            ON CONFLICT REPLACE
            """
            )
        ]


class TranslationCache:
    @staticmethod
    def _sort_dict_recursively(obj):
        if isinstance(obj, dict):
            return {
                k: TranslationCache._sort_dict_recursively(v)
                for k in sorted(obj.keys())
                for v in [obj[k]]
            }
        elif isinstance(obj, list):
            return [TranslationCache._sort_dict_recursively(item) for item in obj]
        return obj

    def __init__(self, translate_engine: str, translate_engine_params: dict = None):
        assert len(translate_engine) < 20, (
            "current cache require translate engine name less than 20 characters"
        )
        self.translate_engine = translate_engine
        self.replace_params(translate_engine_params)

    # The program typically starts multi-threaded translation
    # only after cache parameters are fully configured,
    # so thread safety doesn't need to be considered here.
    def replace_params(self, params: dict = None):
        if params is None:
            params = {}
        self.params = params
        params = self._sort_dict_recursively(params)
        self.translate_engine_params = json.dumps(params)

    def update_params(self, params: dict = None):
        if params is None:
            params = {}
        self.params.update(params)
        self.replace_params(self.params)

    def add_params(self, k: str, v):
        self.params[k] = v
        self.replace_params(self.params)

    # Since peewee and the underlying sqlite are thread-safe,
    # get and set operations don't need locks.
    def get(self, original_text: str) -> str | None:
        result = _TranslationCache.get_or_none(
            translate_engine=self.translate_engine,
            translate_engine_params=self.translate_engine_params,
            original_text=original_text,
        )
        return result.translation if result else None

    def set(self, original_text: str, translation: str):
        try:
            _TranslationCache.create(
                translate_engine=self.translate_engine,
                translate_engine_params=self.translate_engine_params,
                original_text=original_text,
                translation=translation,
            )
        except Exception as e:
            logger.debug(f"Error setting cache: {e}")


def init_db(remove_exists=False):
    cache_folder = Path.home() / ".cache" / "pdf2zh.kernel.experimental"
    cache_folder.mkdir(parents=True, exist_ok=True)
    # The current version does not support database migration, so add the version number to the file name.
    cache_db_path = cache_folder / "cache.v1.db"
    if remove_exists and cache_db_path.exists():
        cache_db_path.unlink()
    db.init(
        str(cache_db_path),
        pragmas={
            "journal_mode": "wal",
            "busy_timeout": 1000,
        },
    )
    db.create_tables([_TranslationCache], safe=True)


def init_test_db():
    import os
    import tempfile

    fd, cache_db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)  # Close the file descriptor as we only need the path
    test_db = SqliteDatabase(
        cache_db_path,
        pragmas={
            "journal_mode": "wal",
            "busy_timeout": 1000,
        },
    )
    test_db.bind([_TranslationCache], bind_refs=False, bind_backrefs=False)
    test_db.connect()
    test_db.create_tables([_TranslationCache], safe=True)
    return test_db


def clean_test_db(test_db):
    test_db.drop_tables([_TranslationCache])
    test_db.close()
    db_path = Path(test_db.database)
    if db_path.exists():
        db_path.unlink()
    wal_path = Path(str(db_path) + "-wal")
    if wal_path.exists():
        wal_path.unlink()
    shm_path = Path(str(db_path) + "-shm")
    if shm_path.exists():
        shm_path.unlink()


init_db()

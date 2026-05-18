import os

from dotenv import load_dotenv


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_repo_env(override: bool = True) -> str | None:
    env_path = os.path.join(repo_root(), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=override)
        return env_path
    load_dotenv(override=override)
    return None

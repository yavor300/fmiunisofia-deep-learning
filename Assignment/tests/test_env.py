from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.cityseg.env import load_env_file


class TestLoadEnvFile(unittest.TestCase):
    def test_when_env_file_has_token_then_token_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("HF_TOKEN=abc123\n", encoding="utf-8")

            original = os.environ.pop("HF_TOKEN", None)
            try:
                loaded = load_env_file(env_path)

                self.assertEqual(loaded["HF_TOKEN"], "abc123")
                self.assertEqual(os.environ["HF_TOKEN"], "abc123")
            finally:
                _restore_env("HF_TOKEN", original)

    def test_when_env_var_exists_then_existing_value_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("HF_TOKEN=file-token\n", encoding="utf-8")

            original = os.environ.get("HF_TOKEN")
            os.environ["HF_TOKEN"] = "shell-token"
            try:
                load_env_file(env_path)

                self.assertEqual(os.environ["HF_TOKEN"], "shell-token")
            finally:
                _restore_env("HF_TOKEN", original)

    def test_when_override_is_true_then_existing_value_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("HF_TOKEN=file-token\n", encoding="utf-8")

            original = os.environ.get("HF_TOKEN")
            os.environ["HF_TOKEN"] = "shell-token"
            try:
                load_env_file(env_path, override=True)

                self.assertEqual(os.environ["HF_TOKEN"], "file-token")
            finally:
                _restore_env("HF_TOKEN", original)


def _restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()

"""Keys from the environment, else this computer's store (`fibsem.keystore`),
and ``fibsem-cli key``, against an in-memory store: never the real one."""

from types import SimpleNamespace

import pytest

from fibsem import cli, keystore
from fibsem.alerts import service

KEY = "ANTHROPIC_API_KEY"


class MemoryStore:
    def __init__(self, fail=False):
        self.kept = {}
        self.fail = fail

    def get_password(self, service, name):
        if self.fail:
            raise RuntimeError("locked")
        return self.kept.get((service, name))

    def set_password(self, service, name, value):
        self.kept[(service, name)] = value

    def delete_password(self, service, name):
        del self.kept[(service, name)]


@pytest.fixture
def store(monkeypatch):
    kept = MemoryStore()
    monkeypatch.setattr(keystore, "_store", lambda: kept)
    monkeypatch.setattr(keystore, "sys", SimpleNamespace(platform="win32"))
    for name in keystore.KEYS:
        monkeypatch.delenv(name, raising=False)
    return kept


@pytest.fixture
def run(capsys, monkeypatch):
    """fibsem-cli with *argv*, the key typed at the hidden prompt: (code, out)."""

    def run(*argv, typed=""):
        monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: typed)
        with pytest.raises(SystemExit) as exit:
            cli.main(list(argv))
        out, err = capsys.readouterr()
        return exit.value.code, out + err

    return run


def test_the_environment_wins_then_the_store(store, monkeypatch):
    assert keystore.get_secret(KEY) is None
    keystore.set_secret(KEY, "sk-stored")
    assert keystore.get_secret(KEY) == "sk-stored"
    monkeypatch.setenv(KEY, "sk-env")
    assert keystore.get_secret(KEY) == "sk-env"
    assert keystore.get_secret(KEY, environ={}) == "sk-stored"


def test_only_windows_and_mac_have_a_store(monkeypatch):
    # keyring's other backends include plain-text files.
    monkeypatch.setattr(keystore, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.delenv(KEY, raising=False)
    assert keystore._store() is None
    assert keystore.store_name() is None
    assert keystore.get_secret(KEY) is None
    with pytest.raises(keystore.SecretStoreError, match=KEY):
        keystore.set_secret(KEY, "sk-test")
    assert keystore.delete_secret(KEY) is False


def test_a_store_that_fails_reads_as_no_key(monkeypatch):
    monkeypatch.setattr(keystore, "_store", lambda: MemoryStore(fail=True))
    assert keystore.stored_secret(KEY) is None


def test_key_set_stores_what_was_typed_and_never_prints_it(store, run):
    code, out = run("key", "set", KEY, typed=" sk-typed \n")
    assert code == 0
    assert store.kept[(keystore.SERVICE, KEY)] == "sk-typed"
    assert "Windows Credential Manager" in out
    assert "sk-typed" not in out

    code, out = run("key", "status", KEY)
    assert (code, "sk-typed" in out) == (0, False)
    assert f"{KEY}: stored." in out

    code, out = run("key", "remove", KEY)
    assert code == 0 and "Removed" in out
    assert run("key", "status", KEY)[1].startswith(f"{KEY}: not stored.")
    assert "No " in run("key", "remove", KEY)[1]


def test_key_set_with_nothing_typed_stores_nothing(store, run):
    code, _ = run("key", "set", KEY, typed="   ")
    assert code == 1 and store.kept == {}


def test_key_says_when_the_environment_overrides_the_stored_key(
    store, run, monkeypatch
):
    monkeypatch.setenv(KEY, "sk-env")
    code, out = run("key", "set", KEY, typed="sk-typed")
    assert code == 0 and "used first" in out
    assert "sk-env" not in out and "sk-typed" not in out


def test_key_takes_only_the_keys_fibsem_reads(store, run):
    code, _ = run("key", "set", "SOME_OTHER_KEY", typed="x")
    assert code == 2 and store.kept == {}


def test_key_without_a_store_says_to_use_the_environment(run, monkeypatch):
    monkeypatch.setattr(keystore, "sys", SimpleNamespace(platform="linux"))
    code, out = run("key", "set", KEY, typed="sk-typed")
    assert code == 1 and "environment variable" in out


def test_the_slack_token_can_be_stored(store, monkeypatch):
    assert service.slack_token() == ""
    keystore.set_secret(service.TOKEN_ENV, "xoxb-stored")
    assert service.slack_token() == "xoxb-stored"

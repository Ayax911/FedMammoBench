"""Tests for WandbWriter's degradation ladder.

WandbConfig.enabled defaults to True (configs/wandb_config.py), so a
WandbWriter gets constructed on EVERY training run — including keyless
Docker containers with no WANDB_API_KEY. These tests exist to prove that
every failure mode (not installed, no credentials, network down, wandb.init
raising, wandb.log raising) degrades to "keep training, no W&B data this
run" and never hangs, prompts, or raises out of a public method.

All tests stub sys.modules["wandb"] (via monkeypatch, auto-reverted), so
they pass identically whether or not the real wandb package is installed in
the test environment.

Run::

    pytest tests/test_wandb_writer.py -v
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


class _FakeRun:
    def __init__(self, *, fail_log: bool = False) -> None:
        self.logged: list[tuple[dict, int]] = []
        self.finished = False
        self._fail_log = fail_log

    def log(self, payload: dict, step: int) -> None:
        if self._fail_log:
            raise RuntimeError("simulated network drop")
        self.logged.append((dict(payload), step))

    def finish(self) -> None:
        self.finished = True


def _install_fake_wandb(
    monkeypatch: pytest.MonkeyPatch,
    *,
    api_key: str | None = None,
    init_raises: bool | list[bool] = False,
    fail_log: bool = False,
) -> types.SimpleNamespace:
    """Install a fake ``wandb`` module and return a recorder of init() calls.

    ``init_raises``: either a bool (every init() call raises/doesn't) or a
    list consumed one entry per call (e.g. [True, False] -> first call
    raises, second succeeds — for testing the online-fails/offline-retry path).
    """
    calls: list[dict[str, Any]] = []
    raises_iter = iter(init_raises) if isinstance(init_raises, list) else None
    run_holder: dict[str, _FakeRun] = {}

    def _init(**kwargs: Any) -> _FakeRun:
        calls.append(kwargs)
        should_raise = next(raises_iter) if raises_iter is not None else init_raises
        if should_raise:
            raise RuntimeError("simulated wandb.init failure")
        run = _FakeRun(fail_log=fail_log)
        run_holder["run"] = run
        return run

    fake_module = types.ModuleType("wandb")
    fake_module.init = _init  # type: ignore[attr-defined]
    fake_module.Settings = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    fake_module.api = types.SimpleNamespace(api_key=api_key)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "wandb", fake_module)
    return types.SimpleNamespace(calls=calls, run_holder=run_holder)


@pytest.fixture(autouse=True)
def _clean_wandb_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with a clean slate — no leaked env from the host/CI."""
    for var in ("WANDB_API_KEY", "WANDB_MODE", "WANDB_SILENT", "WANDB_CONSOLE"):
        monkeypatch.delenv(var, raising=False)


class TestDisabled:
    def test_disabled_is_noop_and_never_imports_wandb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__
        import_calls = []

        def _counting_import(name, *args, **kwargs):
            if name == "wandb":
                import_calls.append(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _counting_import)

        from fedmammobench.utils.wandb_utils import WandbWriter

        w = WandbWriter(project="p", enabled=False)
        assert w._run is None
        assert import_calls == []  # wandb must never even be imported

        # Every public method stays a safe no-op.
        w.log_scalar("x", 1.0, 0)
        w.log_scalars("g", {"y": 2.0}, 0)
        w.flush()
        w.close()
        w.close()  # idempotent

    def test_mode_disabled_is_equivalent(self) -> None:
        from fedmammobench.utils.wandb_utils import WandbWriter

        w = WandbWriter(project="p", enabled=True, mode="disabled")
        assert w._run is None


class TestNotInstalled:
    def test_missing_wandb_module_degrades(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(sys.modules, "wandb", raising=False)
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "wandb":
                raise ImportError("simulated: not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        from fedmammobench.utils.wandb_utils import WandbWriter

        w = WandbWriter(project="p", log_dir="/tmp/wb_test_not_installed")
        assert w._run is None
        w.log_scalars("g", {"y": 1.0}, 0)  # no-op, no raise


class TestCredentialDetection:
    def test_no_api_key_forces_offline(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        rec = _install_fake_wandb(monkeypatch, api_key=None)
        from fedmammobench.utils.wandb_utils import WandbWriter

        w = WandbWriter(project="p", mode="online", log_dir=tmp_path)
        assert w._run is not None
        assert rec.calls[0]["mode"] == "offline"

    def test_empty_api_key_string_treated_as_absent(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        # This is exactly what `-e WANDB_API_KEY="${WANDB_API_KEY:-}"` passes
        # into a container when the host variable is unset.
        monkeypatch.setenv("WANDB_API_KEY", "")
        rec = _install_fake_wandb(monkeypatch, api_key=None)
        from fedmammobench.utils.wandb_utils import WandbWriter

        WandbWriter(project="p", mode="online", log_dir=tmp_path)
        assert rec.calls[0]["mode"] == "offline"

    def test_env_api_key_present_stays_online(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "sk-fake-key")
        rec = _install_fake_wandb(monkeypatch, api_key=None)
        from fedmammobench.utils.wandb_utils import WandbWriter

        WandbWriter(project="p", mode="online", log_dir=tmp_path)
        assert rec.calls[0]["mode"] == "online"

    def test_cached_netrc_credentials_stay_online(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        rec = _install_fake_wandb(monkeypatch, api_key="cached-from-netrc")
        from fedmammobench.utils.wandb_utils import WandbWriter

        WandbWriter(project="p", mode="online", log_dir=tmp_path)
        assert rec.calls[0]["mode"] == "online"

    def test_offline_mode_never_checks_credentials(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        rec = _install_fake_wandb(monkeypatch, api_key=None)
        from fedmammobench.utils.wandb_utils import WandbWriter

        WandbWriter(project="p", mode="offline", log_dir=tmp_path)
        assert rec.calls[0]["mode"] == "offline"

    def test_wandb_mode_env_overrides_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "sk-fake-key")
        monkeypatch.setenv("WANDB_MODE", "offline")
        rec = _install_fake_wandb(monkeypatch, api_key=None)
        from fedmammobench.utils.wandb_utils import WandbWriter

        WandbWriter(project="p", mode="online", log_dir=tmp_path)  # config says online
        assert rec.calls[0]["mode"] == "offline"  # env wins


class TestInitFailure:
    def test_init_exception_is_swallowed_then_retried_offline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "sk-fake-key")
        rec = _install_fake_wandb(monkeypatch, init_raises=[True, False])
        from fedmammobench.utils.wandb_utils import WandbWriter

        w = WandbWriter(project="p", mode="online", log_dir=tmp_path)
        assert len(rec.calls) == 2
        assert rec.calls[0]["mode"] == "online"
        assert rec.calls[1]["mode"] == "offline"
        assert w._run is not None  # second attempt succeeded

    def test_both_attempts_fail_disables_cleanly(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "sk-fake-key")
        _install_fake_wandb(monkeypatch, init_raises=True)
        from fedmammobench.utils.wandb_utils import WandbWriter

        w = WandbWriter(project="p", mode="online", log_dir=tmp_path)
        assert w._run is None
        w.log_scalars("g", {"y": 1.0}, 0)  # no-op, no raise
        w.close()  # no-op, no raise


class TestLogging:
    def test_log_scalars_reaches_run_with_prefix(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "sk-fake-key")
        rec = _install_fake_wandb(monkeypatch)
        from fedmammobench.utils.wandb_utils import WandbWriter

        w = WandbWriter(project="p", mode="online", log_dir=tmp_path)
        w.log_scalars("train", {"loss": 0.5, "auc": 0.9}, step=3)
        payload, step = rec.run_holder["run"].logged[0]
        assert payload == {"train/loss": 0.5, "train/auc": 0.9}
        assert step == 3

    def test_non_numeric_metric_skipped_not_raised(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "sk-fake-key")
        rec = _install_fake_wandb(monkeypatch)
        from fedmammobench.utils.wandb_utils import WandbWriter

        w = WandbWriter(project="p", mode="online", log_dir=tmp_path)
        w.log_scalars("g", {"loss": 0.5, "note": "not a number"}, step=0)
        payload, _ = rec.run_holder["run"].logged[0]
        assert payload == {"g/loss": 0.5}

    def test_log_failure_disables_after_max_without_raising(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "sk-fake-key")
        _install_fake_wandb(monkeypatch, fail_log=True)
        from fedmammobench.utils.wandb_utils import _MAX_LOG_FAILURES, WandbWriter

        w = WandbWriter(project="p", mode="online", log_dir=tmp_path)
        for step in range(_MAX_LOG_FAILURES + 2):
            w.log_scalars("g", {"y": 1.0}, step)  # must never raise
        assert w._run is None  # disabled itself after _MAX_LOG_FAILURES

    def test_close_calls_finish_and_is_idempotent(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "sk-fake-key")
        rec = _install_fake_wandb(monkeypatch)
        from fedmammobench.utils.wandb_utils import WandbWriter

        w = WandbWriter(project="p", mode="online", log_dir=tmp_path)
        run = rec.run_holder["run"]
        w.close()
        assert run.finished is True
        assert w._run is None
        w.close()  # second call must not raise (no run left to finish)


class TestConsoleAndSilentEnv:
    def test_console_and_silent_forced_off_before_import(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        # Not set beforehand (see _clean_wandb_env) -> WandbWriter must set them.
        monkeypatch.setenv("WANDB_API_KEY", "sk-fake-key")
        _install_fake_wandb(monkeypatch)
        from fedmammobench.utils.wandb_utils import WandbWriter

        WandbWriter(project="p", mode="online", log_dir=tmp_path)
        assert __import__("os").environ["WANDB_CONSOLE"] == "off"
        assert __import__("os").environ["WANDB_SILENT"] == "true"

    def test_operator_override_is_respected(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        # setdefault semantics: an operator's explicit choice must win.
        monkeypatch.setenv("WANDB_CONSOLE", "auto")
        monkeypatch.setenv("WANDB_API_KEY", "sk-fake-key")
        _install_fake_wandb(monkeypatch)
        from fedmammobench.utils.wandb_utils import WandbWriter

        WandbWriter(project="p", mode="online", log_dir=tmp_path)
        assert __import__("os").environ["WANDB_CONSOLE"] == "auto"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-v"]))

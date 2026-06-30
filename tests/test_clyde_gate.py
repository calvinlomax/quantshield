from __future__ import annotations

import sys
import types

import pytest

import quantshield_app.main as app_main
from quantshield_app.clyde_gate import (
    CLYDE_CODE_ENV_VAR,
    PINNED_CLYDE_CODE,
    image_code,
    validate_clyde_runtime_environment,
)


def test_image_code_is_stable_for_the_same_image(tmp_path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color=(12, 34, 56)).save(image_path)

    first_code = image_code(image_path)
    second_code = image_code(image_path)

    assert first_code == second_code
    assert len(first_code) == 32


def test_image_code_changes_when_image_changes(tmp_path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color=(12, 34, 56)).save(image_path)
    first_code = image_code(image_path)

    Image.new("RGB", (32, 32), color=(12, 34, 57)).save(image_path)
    second_code = image_code(image_path)

    assert first_code != second_code


def test_validate_clyde_runtime_environment_requires_expected_env_code(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "clyde.jpg"
    image_path.write_bytes(b"placeholder")
    monkeypatch.setattr("quantshield_app.clyde_gate.image_code", lambda _path: PINNED_CLYDE_CODE)

    with pytest.raises(SystemExit, match=CLYDE_CODE_ENV_VAR):
        validate_clyde_runtime_environment(env={}, image_path=image_path)


def test_validate_clyde_runtime_environment_rejects_unpinned_asset_code(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "clyde.jpg"
    image_path.write_bytes(b"placeholder")
    monkeypatch.setattr("quantshield_app.clyde_gate.image_code", lambda _path: "QdQVKK276ugizr9d0RMRTmHgOQ11saOl")

    with pytest.raises(SystemExit, match="currently resolves to"):
        validate_clyde_runtime_environment(
            env={CLYDE_CODE_ENV_VAR: PINNED_CLYDE_CODE},
            image_path=image_path,
        )


def test_validate_clyde_runtime_environment_accepts_matching_pin(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "clyde.jpg"
    image_path.write_bytes(b"placeholder")
    monkeypatch.setattr("quantshield_app.clyde_gate.image_code", lambda _path: PINNED_CLYDE_CODE)

    assert (
        validate_clyde_runtime_environment(
            env={CLYDE_CODE_ENV_VAR: PINNED_CLYDE_CODE},
            image_path=image_path,
        )
        == PINNED_CLYDE_CODE
    )


def test_main_blocks_startup_when_clyde_gate_fails(monkeypatch) -> None:
    def _raise_blocked() -> None:
        raise SystemExit("blocked")

    monkeypatch.setattr(app_main, "validate_clyde_runtime_environment", _raise_blocked)

    with pytest.raises(SystemExit, match="blocked"):
        app_main.main([])


def test_main_launches_when_clyde_gate_passes(monkeypatch) -> None:
    monkeypatch.setattr(app_main, "validate_clyde_runtime_environment", lambda: None)

    class FakeQApplication:
        def __init__(self, argv: list[str]) -> None:
            self.argv = argv

        def exec(self) -> int:
            return 27

    checkpoint_roots: list[list[str] | None] = []
    shown_windows: list[object] = []

    class FakeCheckpointService:
        def __init__(self, search_roots: list[str] | None = None) -> None:
            checkpoint_roots.append(search_roots)

    class FakeWindow:
        def __init__(self, checkpoint_service: FakeCheckpointService) -> None:
            self.checkpoint_service = checkpoint_service

        def show(self) -> None:
            shown_windows.append(self)

    qtwidgets_module = types.ModuleType("PySide6.QtWidgets")
    qtwidgets_module.QApplication = FakeQApplication
    monkeypatch.setitem(sys.modules, "PySide6", types.ModuleType("PySide6"))
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", qtwidgets_module)

    services_module = types.ModuleType("quantshield_app.services")
    services_module.CheckpointService = FakeCheckpointService
    monkeypatch.setitem(sys.modules, "quantshield_app.services", services_module)

    ui_module = types.ModuleType("quantshield_app.ui")
    ui_module.QuantShieldMainWindow = FakeWindow
    monkeypatch.setitem(sys.modules, "quantshield_app.ui", ui_module)

    assert app_main.main(["--checkpoint-root", "alpha", "--checkpoint-root", "beta"]) == 27
    assert checkpoint_roots == [["alpha", "beta"]]
    assert len(shown_windows) == 1

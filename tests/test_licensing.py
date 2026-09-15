# SPDX-License-Identifier: Apache-2.0
"""License lookup must work without depending on the repository working directory."""

import importlib.metadata

import pytest

from entelechy import licensing


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    root = tmp_path / "source"
    package = root / "entelechy"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "entelechy-kernels"\n', encoding="utf-8"
    )
    monkeypatch.setattr(licensing, "__file__", str(package / "licensing.py"))

    def not_installed(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", not_installed)
    return root


@pytest.fixture
def installed_wheel(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    package = site / "entelechy"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    info = site / "entelechy_kernels-0.0.1.dist-info"
    (info / "licenses").mkdir(parents=True)
    (info / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: entelechy-kernels\nVersion: 0.0.1\n"
        "License-Expression: Apache-2.0\nLicense-File: LICENSE\n",
        encoding="utf-8",
    )
    (info / "RECORD").write_text(
        f"{info.name}/licenses/LICENSE,,\n{info.name}/METADATA,,\n",
        encoding="utf-8",
    )
    distribution = importlib.metadata.Distribution.at(info)

    def installed(name):
        assert name == "entelechy-kernels"
        return distribution

    monkeypatch.setattr(importlib.metadata, "distribution", installed)
    monkeypatch.setattr(licensing, "__file__", str(package / "licensing.py"))
    return site, info


def test_read_license_from_uninstalled_source_tree(source_tree, tmp_path, monkeypatch):
    text = "Entelechy source license\n\nFull license text.\n"
    (source_tree / "LICENSE").write_text(text, encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "LICENSE").write_text("Another project's license", encoding="utf-8")
    monkeypatch.chdir(elsewhere)
    assert licensing.project_license_text() == text


def test_read_license_from_installed_wheel(installed_wheel):
    site, info = installed_wheel
    text = "Entelechy wheel license\n\nFull license text.\n"
    (info / "licenses" / "LICENSE").write_text(text, encoding="utf-8")
    (site / "LICENSE").write_text("Another project's license", encoding="utf-8")
    assert licensing.project_license_text() == text


@pytest.mark.parametrize("contents", [None, "", " \n\t"])
def test_missing_or_empty_source_license_fails(source_tree, contents):
    if contents is not None:
        (source_tree / "LICENSE").write_text(contents, encoding="utf-8")
    with pytest.raises(RuntimeError, match="[Ll]icense"):
        licensing.project_license_text()


def test_missing_wheel_license_does_not_use_unrelated_license(installed_wheel):
    site, _ = installed_wheel
    (site / "LICENSE").write_text("Another project's license", encoding="utf-8")
    with pytest.raises(RuntimeError, match="(?i)license"):
        licensing.project_license_text()


@pytest.mark.parametrize("record", ["", "LICENSE,,\n", "other-1.dist-info/licenses/LICENSE,,\n"])
def test_only_own_recorded_license_is_used(installed_wheel, record):
    site, info = installed_wheel
    (info / "RECORD").write_text(record, encoding="utf-8")
    (site / "LICENSE").write_text("Another project's license", encoding="utf-8")
    other = site / "other-1.dist-info" / "licenses"
    other.mkdir(parents=True)
    (other / "LICENSE").write_text("Another project's license", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not contain its declared LICENSE"):
        licensing.project_license_text()


def test_uninstalled_module_without_source_layout_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(licensing, "__file__", str(tmp_path / "entelechy" / "licensing.py"))
    (tmp_path / "LICENSE").write_text("Another project's license", encoding="utf-8")

    def not_installed(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", not_installed)
    with pytest.raises(RuntimeError, match="Cannot locate"):
        licensing.project_license_text()

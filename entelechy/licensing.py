# SPDX-License-Identifier: Apache-2.0
"""Find this project's full license for standalone kernel exports."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path


def _read_license(path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"Cannot read the Entelechy project license: {path}") from error
    if not text.strip():
        raise RuntimeError(f"The Entelechy project license is empty: {path}")
    return text


def project_license_text() -> str:
    """Read LICENSE from this source tree or this distribution's wheel metadata.

    Source checkouts and editable installations use their own repository license.
    Wheel installations use the PEP 639 license file recorded by the Entelechy
    distribution. Never search the working directory or a generic site-packages
    LICENSE, which could belong to another project. Missing license data prevents
    an incomplete export rather than silently omitting the license.
    """
    package_dir = Path(__file__).resolve().parent
    source_root = package_dir.parent
    if (
        package_dir.name == "entelechy"
        and (source_root / "pyproject.toml").is_file()
        and (package_dir / "__init__.py").is_file()
    ):
        return _read_license(source_root / "LICENSE")

    try:
        distribution = importlib.metadata.distribution("entelechy-kernels")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Cannot locate the Entelechy project license for export") from error

    declared_files = distribution.metadata.get_all("License-File") or []
    expected = (
        f"entelechy_kernels-{distribution.version}.dist-info",
        "licenses",
        "LICENSE",
    )
    if "LICENSE" in declared_files:
        for entry in distribution.files or ():
            if entry.parts == expected:
                return _read_license(distribution.locate_file(entry))
    raise RuntimeError(
        "The entelechy-kernels distribution does not contain its declared LICENSE; "
        "reinstall a complete package before exporting kernels"
    )

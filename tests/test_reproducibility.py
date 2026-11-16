"""CAL-11: the reproducible install must pin and hash the full dependency graph."""
from importlib import metadata
from pathlib import Path
import re
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def test_cal_11_lock_covers_runtime_test_and_build_dependencies():
    root = Path(__file__).resolve().parents[1]
    lock = root / "requirements.lock"
    assert lock.is_file(), "missing hash-bearing dependency lock"
    records = lock.read_text().replace("\\\n", " ").splitlines()
    locked = {}
    for record in records:
        if not record.strip() or record.lstrip().startswith("#"):
            continue
        requirement, *hashes = record.split("--hash=")
        req = Requirement(requirement.strip())
        assert re.fullmatch(r"==[^*,\s]+", str(req.specifier)), requirement
        assert hashes and all(re.fullmatch(r"sha256:[0-9a-f]{64}", h.strip()) for h in hashes)
        if req.marker is None or req.marker.evaluate():
            locked[canonicalize_name(req.name)] = req

    project = tomllib.loads((root / "pyproject.toml").read_text())
    build = [line for line in (root / "requirements-build.in").read_text().splitlines()
             if line and not line.startswith("#")]
    assert build == project["build-system"]["requires"]
    declared = (project["project"]["dependencies"]
                + project["project"]["optional-dependencies"]["test"]
                + project["build-system"]["requires"])
    for requirement in declared:
        req = Requirement(requirement)
        name = canonicalize_name(req.name)
        assert name in locked, f"unlocked declared dependency: {name}"
        version = next(iter(locked[name].specifier)).version
        assert version in req.specifier

    # Installed metadata provides an independent check of transitive completeness.
    for name, pin in locked.items():
        assert metadata.version(name) in pin.specifier, f"install the lock for {name}"
        for requirement in metadata.requires(name) or []:
            req = Requirement(requirement)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            dependency = canonicalize_name(req.name)
            assert dependency in locked, f"unlocked dependency of {name}: {dependency}"
            assert metadata.version(dependency) in req.specifier

"""Export installed versions from original wheels or verified uv wheel archives.

This performs no downloads, installs, or cache cleanup. Repacked wheels have new
local SHA256 values; upstream wheel hashes remain separate provenance references.
The default source is the project's .venv; use --environment for another one.
An existing wheelhouse plus this provenance report is sufficient for reruns;
the original uv cache and historical environment directories are optional.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import email
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tomllib
import urllib.parse
import zipfile

from packaging.tags import parse_tag, sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / ".cache/offline/wheels"
REPORT = ROOT / ".cache/offline/provenance/export.json"
SUPPORTED_TAGS = set(sys_tags())


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).digest()


def relative(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def installed(environment):
    sites = sorted(environment.glob("lib/python*/site-packages"))
    if len(sites) != 1:
        raise RuntimeError(f"Expected one Python site-packages directory: {environment}")
    result = {}
    for distribution in importlib.metadata.distributions(path=[str(sites[0])]):
        name = canonicalize_name(distribution.metadata["Name"])
        result[name] = {
            "name": name, "version": distribution.version,
            "installed_metadata": relative(Path(distribution._path)),
        }
    return result


def scan_wheels():
    result = {}
    # Existing offline wheels may include independently built compatibility wheels.
    roots = [DEST, ROOT / ".cache/wheels", ROOT / ".cache/verified-wheels",
             ROOT / ".cache/uv/sdists-v9", ROOT / ".dev/artifacts"]
    for root in roots:
        for path in sorted(root.rglob("*.whl")) if root.exists() else []:
            try:
                name, version, _, tags = parse_wheel_filename(path.name)
            except ValueError:
                continue
            if tags & SUPPORTED_TAGS:
                result.setdefault((name, str(version)), []).append(path)
    return result


def scan_archives():
    result = {}
    cache = ROOT / ".cache/uv/archive-v0"
    for archive in sorted(cache.iterdir()) if cache.exists() else []:
        if not archive.is_dir():
            continue
        for info in sorted(archive.glob("*.dist-info")):
            metadata = info / "METADATA"
            if not metadata.is_file():
                continue
            message = email.message_from_string(metadata.read_text())
            key = (canonicalize_name(message["Name"]), message["Version"])
            result.setdefault(key, []).append((archive, info))
    return result


def upstream_references():
    result = {}
    locks = [ROOT / "uv.lock", *sorted((ROOT / "environments").glob("*/uv.lock"))]
    for path in locks:
        if not path.exists():
            continue
        for package in tomllib.loads(path.read_text()).get("package", []):
            for wheel in package.get("wheels", []):
                url = wheel.get("url")
                if not url:
                    continue
                filename = urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit("/", 1)[-1])
                key = (canonicalize_name(package["name"]), package["version"], filename)
                reference = {"url": url, "upstream_sha256": wheel.get("hash", "").removeprefix("sha256:") or None,
                             "recorded_in": relative(path), "source": "uv.lock"}
                result.setdefault(key, []).append(reference)
    for path in sorted((ROOT / "environments").glob("*/*wheels.json")):
        for wheel in json.loads(path.read_text()).get("wheels", []):
            key = (canonicalize_name(wheel["name"]), wheel["version"], wheel["filename"])
            result.setdefault(key, []).append({
                "url": wheel["url"], "upstream_sha256": wheel["sha256"],
                "recorded_in": relative(path),
                "source": wheel.get("sha256_source", "project manifest pinned to upstream artifact"),
            })
    return result


def validate_wheel_identity(path, expected):
    with zipfile.ZipFile(path) as archive:
        metadata_paths = [name for name in archive.namelist()
                          if name.count("/") == 1 and name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise RuntimeError(f"Expected one top-level wheel METADATA: {path}")
        metadata = email.message_from_bytes(archive.read(metadata_paths[0]))
        actual = (canonicalize_name(metadata["Name"]), metadata["Version"])
        if actual != expected:
            raise RuntimeError(f"Wheel METADATA mismatch: {path}: {actual} != {expected}")


def archive_filename(info, package, references):
    wheel = email.message_from_string((info / "WHEEL").read_text())
    tags = set().union(*(parse_tag(tag) for tag in wheel.get_all("Tag", [])))
    if not tags or not tags & SUPPORTED_TAGS:
        raise RuntimeError(f"No compatible wheel tags: {info}")
    # Prefer the original filename when its full tag set matches the archive.
    for name, version, filename in references:
        if (name, version) != (package["name"], package["version"]):
            continue
        if parse_wheel_filename(filename)[3] == tags:
            return filename
    interpreters = ".".join(sorted({tag.interpreter for tag in tags}))
    abis = ".".join(sorted({tag.abi for tag in tags}))
    platforms = ".".join(sorted({tag.platform for tag in tags}))
    combined = f"{interpreters}-{abis}-{platforms}"
    if parse_tag(combined) != tags:
        raise RuntimeError(f"Cannot losslessly compress wheel tags: {info}")
    build = f"-{wheel['Build']}" if wheel.get("Build") else ""
    return f"{package['name'].replace('-', '_')}-{package['version']}{build}-{combined}.whl"


def verified_archive_files(archive, info):
    record_path = info / "RECORD"
    record_name = record_path.relative_to(archive).as_posix()
    files = []
    excluded_signatures = []
    seen = set()
    with record_path.open(newline="") as stream:
        for row in csv.reader(stream):
            if len(row) != 3:
                raise RuntimeError(f"Malformed RECORD row: {row}")
            name, expected_hash, expected_size = row
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or name in seen:
                raise RuntimeError(f"Unsafe or duplicate RECORD path: {name}")
            seen.add(name)
            source = archive / name
            if source.is_symlink() or not source.is_file():
                raise RuntimeError(f"Missing/nonregular RECORD file: {source}")
            if name == record_name:
                if expected_hash or expected_size:
                    raise RuntimeError(f"RECORD must have empty hash/size for itself: {source}")
                continue
            if name in {record_name + ".jws", record_name + ".p7s"}:
                excluded_signatures.append(name)
                continue
            if not expected_hash or not expected_size:
                raise RuntimeError(f"RECORD file lacks checksum or size: {source}")
            algorithm, encoded = expected_hash.split("=", 1)
            if algorithm not in {"sha256", "sha384", "sha512"}:
                raise RuntimeError(f"Unsupported RECORD hash algorithm: {algorithm}")
            actual_size = source.stat().st_size
            actual_hash = base64.urlsafe_b64encode(digest(source, algorithm)).rstrip(b"=").decode()
            if actual_size != int(expected_size) or actual_hash != encoded:
                raise RuntimeError(f"RECORD checksum/size mismatch: {source}")
            files.append((name, source, expected_hash, str(actual_size)))
    if record_name not in seen:
        raise RuntimeError(f"RECORD does not list itself: {record_path}")
    return sorted(files), record_name, excluded_signatures


def zip_info(name, mode):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | stat.S_IMODE(mode)) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    info._compresslevel = 6
    return info


def repack(archive, info, filename):
    files, record_name, excluded = verified_archive_files(archive, info)
    destination = DEST / filename
    temporary = destination.with_suffix(".export-partial")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as wheel:
            rows = []
            for name, source, file_hash, size in files:
                with source.open("rb") as incoming, wheel.open(zip_info(name, source.stat().st_mode), "w") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
                rows.append((name, file_hash, size))
            record = io.StringIO(newline="")
            writer = csv.writer(record, lineterminator="\n")
            writer.writerows([*rows, (record_name, "", "")])
            wheel.writestr(zip_info(record_name, (info / "RECORD").stat().st_mode), record.getvalue().encode())
        validate_wheel_identity(temporary, (canonicalize_name(email.message_from_string((info / "METADATA").read_text())["Name"]),
                                           email.message_from_string((info / "METADATA").read_text())["Version"]))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    native_any = filename.endswith("-any.whl") and any(name.endswith(".so") for name, *_ in files)
    return destination, {"verified_record_entries": len(files), "excluded_record_signatures": excluded,
                         "native_libraries_despite_any_tag": native_any,
                         "deterministic_zip_timestamp": "1980-01-01T00:00:00", "compression": "deflate-6"}


def export_one(package, wheels, archives, references, previous):
    key = (package["name"], package["version"])
    record = dict(package)
    if key in wheels:
        source = wheels[key][0]
        validate_wheel_identity(source, key)
        destination = DEST / source.name
        source_hash = digest(source).hex()
        refs = references.get((*key, source.name), [])
        old = previous.get(key)
        reused_export = source.parent == DEST and old and old.get("local_sha256") == source_hash
        if reused_export:
            # Historical lockfiles may have been removed after consolidation.
            # Keep their references and the original RECORD verification evidence.
            refs = [*old.get("upstream_references", []), *[entry for entry in refs if entry not in old.get("upstream_references", [])]]
        expected = {entry["upstream_sha256"] for entry in refs if entry["upstream_sha256"]}
        known_repack = reused_export and old.get("method") == "repacked-verified-uv-archive"
        if expected and source_hash not in expected and not known_repack:
            raise RuntimeError(f"Existing wheel disagrees with recorded upstream SHA256: {source}")
        if source != destination:
            if destination.exists():
                if digest(destination).hex() != source_hash:
                    raise RuntimeError(f"Refusing to overwrite different wheel: {destination}")
            else:
                try:
                    os.link(source, destination)
                except FileExistsError:
                    if digest(destination).hex() != source_hash:
                        raise RuntimeError(f"Concurrent export created a different wheel: {destination}")
                except OSError:
                    # Exclusive creation also avoids overwriting another exporter.
                    with source.open("rb") as incoming, destination.open("xb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
                    shutil.copystat(source, destination)
        record.update(method="existing-complete-wheel", source=relative(source),
                      upstream_hash_matched=bool(expected and source_hash in expected))
        if reused_export:
            record.update({k: v for k, v in old.items() if k not in {"installed_metadata"}})
            record["reused_existing_export"] = True
    elif key in archives:
        archive, info = archives[key][0]
        filename = archive_filename(info, package, references)
        destination, details = repack(archive, info, filename)
        refs = references.get((*key, filename), [])
        source_hash = digest(destination).hex()
        record.update(method="repacked-verified-uv-archive", source=relative(archive),
                      upstream_hash_matched=False, **details)
    else:
        raise RuntimeError(f"No compatible complete wheel or original uv archive for {key}")
    record.update(filename=destination.name, output=relative(destination), size=destination.stat().st_size,
                  local_sha256=source_hash, upstream_references=refs)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, default=ROOT / ".venv")
    parser.add_argument("--extra-environment", type=Path)
    parser.add_argument("--extra-package", action="append", default=[])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    packages = installed(args.environment.resolve())
    if args.extra_package:
        if not args.extra_environment:
            parser.error("--extra-package requires --extra-environment")
        extra = installed(args.extra_environment.resolve())
        for raw_name in args.extra_package:
            name = canonicalize_name(raw_name)
            if name not in extra:
                parser.error(f"Extra distribution is not installed: {name}")
            if name in packages and packages[name]["version"] != extra[name]["version"]:
                parser.error(f"Extra distribution conflicts with selected environment: {name}")
            packages[name] = extra[name]
    DEST.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    if REPORT.exists():
        previous = {(p["name"], p["version"]): p for p in json.loads(REPORT.read_text()).get("packages", [])}
    wheels, archives, references = scan_wheels(), scan_archives(), upstream_references()
    print(f"Exporting {len(packages)} selected distributions without downloads", flush=True)
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as pool:
        futures = {pool.submit(export_one, package, wheels, archives, references, previous): package
                   for package in packages.values()}
        for future in as_completed(futures):
            package = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"{len(results) + len(failures)}/{len(packages)} {package['name']} {result['method']}", flush=True)
            except Exception as error:
                failures.append({**package, "error": str(error)})
                print(f"FAILED {package['name']}: {error}", flush=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if failures else "passed",
        "source_environment": relative(args.environment.resolve()),
        "target": "CPython 3.11, Linux x86_64; native/system requirements are not supplied by these wheels",
        "scope": "Exact selected installed versions. No network, installs, or cache cleanup.",
        "repack_note": "Local SHA256 identifies the exported wheel. Repacked archives do not reproduce upstream ZIP bytes or upstream wheel SHA256; each original RECORD file hash and size was verified before repacking.",
        "selected_count": len(packages), "exported_count": len(results),
        "packages": sorted(results, key=lambda p: p["name"]),
        "failures": sorted(failures, key=lambda p: p["name"]),
    }
    temporary = REPORT.with_suffix(".export-partial")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(REPORT)
    print(f"Exported {len(results)}/{len(packages)}; provenance: {REPORT}", flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())

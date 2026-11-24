from __future__ import annotations

import argparse
import hashlib
import json
import platform
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

FIXED = (
    "scripts/diagnose_events.py", "scripts/extract_anomaly_features.py",
    "scripts/extract_combined_features.py", "scripts/extract_features.py",
    "scripts/train_quality.py", "scripts/quality_experiment.py", "scripts/independent_event_model.py",
    "contracts/requirements.txt", "contracts/README.md",
    "services/task3_quality_service/README.md", "services/task3_quality_service/requirements.txt",
)
PATTERNS = (
    "services/task3_quality_service/*.py", "shared_contracts/*.py",
    "contracts/schemas/*.json", "contracts/examples/*.json",
)
NOTICE = """# 任务三源码完整性交付

此包是source-only，不是完整运行包；不含权重、视频、许可资料、数据库或离线依赖。
服务仍仅loopback、注册videoId、单进程，候选必须显式opt-in。
复制模型到部署路径后必须提供事先核对的期望SHA-256，不能由收到文件自证可信。
dependencies、可信权重、固定10秒工程夹具、独立环境启动/HTTP验收仍需第二阶段。
本包的导入检查不能代表精度、实时性、HTTP P95或8小时稳定性通过。
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dump_new(path, document):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or ".." in path.parts or path.as_posix() != name or name.endswith("/")):
        raise ValueError("unsafe archive path")
    return path


def allowed_source(name):
    path = PurePosixPath(name)
    return name in FIXED or any(
        path.parent == PurePosixPath(pattern).parent and path.match(pattern)
        for pattern in PATTERNS
    )


def source_files(root):
    names = set(FIXED)
    for pattern in PATTERNS:
        names.update(path.relative_to(root).as_posix() for path in root.glob(pattern) if path.is_file())
    result = {}
    for name in sorted(names):
        safe_name(name)
        path = root / name
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root)
                or any(parent.is_symlink() for parent in path.parents if parent != root)):
            raise ValueError(f"missing or redirected source: {name}")
        result[name] = path.read_bytes()
    if not {"services/task3_quality_service/app.py", "services/task3_quality_service/inference.py",
            "shared_contracts/quality.py"}.issubset(result):
        raise ValueError("required service source is missing")
    return result


def git_value(root, *arguments):
    result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    allowed_output = root / "data/raw/task3-deliveries"
    if not output.is_relative_to(allowed_output) or output == allowed_output:
        raise ValueError("output must be a new task3-deliveries child directory")
    if output.exists():
        raise FileExistsError(output)
    entries = source_files(root)
    entries["README_SOURCE_ONLY.md"] = NOTICE.encode("utf-8")
    manifest = {
        "schemaVersion": "task3.source-delivery/1.0", "deliveryType": "source-only",
        "files": {name: {"sha256": sha(content), "bytes": len(content)} for name, content in entries.items()},
        "missingForRunnableDelivery": ["trusted model sidecar", "platform dependency closure and wheels",
                                       "10-second engineering fixture", "clean-environment startup and HTTP evidence"],
    }
    output.mkdir(parents=True, exist_ok=False)
    archive_path = output / "source-package.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(entries.items()):
            archive.writestr(name, content)
        manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        archive.writestr("MANIFEST.json", manifest_bytes)
    current = source_files(root)
    if {name: sha(value) for name, value in current.items()} != {
        name: sha(value) for name, value in entries.items() if name != "README_SOURCE_ONLY.md"
    }:
        raise ValueError("source changed during build; retained package is unaccepted")
    receipt = {
        "archiveSha256": sha(archive_path.read_bytes()), "manifestSha256": sha(manifest_bytes),
        "payloadFileCount": len(entries), "gitHead": git_value(root, "rev-parse", "HEAD"),
        "gitStatus": git_value(root, "status", "--short"), "python": platform.python_version(),
        "platform": platform.platform(), "deliveryType": "source-only",
    }
    dump_new(output / "receipt.json", receipt)
    return receipt


def verify(archive_path, expected_sha256, destination):
    archive_path, destination = Path(archive_path), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if sha(archive_path.read_bytes()) != expected_sha256:
        raise ValueError("archive hash mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        for info in infos:
            safe_name(info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("archive symlink is not allowed")
        if len(names) != len(set(name.casefold() for name in names)):
            raise ValueError("duplicate archive entry")
        if "MANIFEST.json" not in names:
            raise ValueError("manifest missing")
        manifest = json.loads(archive.read("MANIFEST.json"))
        if manifest.get("schemaVersion") != "task3.source-delivery/1.0" or manifest.get("deliveryType") != "source-only":
            raise ValueError("unsupported delivery manifest")
        files = manifest["files"]
        if set(names) != set(files) | {"MANIFEST.json"}:
            raise ValueError("entry set mismatch")
        if not (set(FIXED) | {"README_SOURCE_ONLY.md", "services/task3_quality_service/app.py",
                             "services/task3_quality_service/inference.py", "shared_contracts/quality.py"}).issubset(files):
            raise ValueError("required source entry missing")
        payload = {}
        for name, expected in files.items():
            if name != "README_SOURCE_ONLY.md" and not allowed_source(name):
                raise ValueError("non-allowlisted entry")
            content = archive.read(name)
            if sha(content) != expected["sha256"] or len(content) != expected["bytes"]:
                raise ValueError("payload hash or size mismatch")
            payload[name] = content
        payload["MANIFEST.json"] = archive.read("MANIFEST.json")
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in payload.items():
        target = destination.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(content)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build/verify a source-only Task 3 delivery")
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--root", type=Path, required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--archive", type=Path, required=True)
    verify_parser.add_argument("--expected-sha256", required=True)
    verify_parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.root, args.output_dir)
    else:
        manifest = verify(args.archive, args.expected_sha256, args.destination)
        result = {"verified": True, "payloadFileCount": len(manifest["files"]), "deliveryType": "source-only"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


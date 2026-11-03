#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


# =========================
# 配置
# =========================

DEFAULT_SESSION = "llm-eval"
REMOTE_ROOT = "/content/llm-post-training"
REMOTE_ARCHIVE = "/content/llm-post-training-source.tar.gz"

# 不上传到 Colab 的目录
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",

    # 实验生成文件
    "wandb",
    "outputs",
    "output",
    "checkpoints",
    "checkpoint",
    "logs",

    # 数据集通常让 Colab 自己下载
    "data",
    "datasets",
}

# 不上传的文件后缀
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
}


def run(cmd: list[str], *, check=True):
    """运行本地命令，并实时显示 stdout/stderr。"""
    print()
    print(">", " ".join(cmd))
    print("-" * 80)

    return subprocess.run(
        cmd,
        check=check,
    )


def should_exclude(path: Path, project_root: Path) -> bool:
    relative = path.relative_to(project_root)

    # 任意一级目录命中排除名单
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return True

    if path.suffix in EXCLUDED_SUFFIXES:
        return True

    return False


def create_archive(project_root: Path, archive_path: Path):
    print(f"[1/4] 打包源码")
    print(f"      项目目录: {project_root}")
    print(f"      输出文件: {archive_path}")

    file_count = 0

    with tarfile.open(archive_path, "w:gz") as tar:
        for directory, dirnames, filenames in os.walk(project_root):
            directory = Path(directory)
            # 直接跳过 outputs 等排除目录，不遍历其中的权重和检查点。
            dirnames[:] = [
                name for name in dirnames
                if not should_exclude(directory / name, project_root)
            ]

            for filename in filenames:
                path = directory / filename
                if should_exclude(path, project_root) or not path.is_file():
                    continue

                relative = path.relative_to(project_root)

                tar.add(
                    path,
                    arcname=relative,
                    recursive=False,
                )

                file_count += 1

    size_mb = archive_path.stat().st_size / 1024 / 1024

    print(f"      文件数量: {file_count}")
    print(f"      压缩包大小: {size_mb:.2f} MB")
    print("      ✓ 打包完成")


def upload_archive(session: str, archive_path: Path):
    print()
    print("[2/4] 上传源码到 Colab")

    run([
        "colab",
        "upload",
        "-s",
        session,
        str(archive_path),
        REMOTE_ARCHIVE,
    ])

    print("      ✓ 上传完成")


def create_remote_setup_script(path: Path):
    script = f"""
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


REMOTE_ROOT = Path("{REMOTE_ROOT}")
ARCHIVE = Path("{REMOTE_ARCHIVE}")


def run(cmd, cwd=None, env=None):
    print()
    print("$", " ".join(map(str, cmd)), flush=True)
    print("-" * 80, flush=True)

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=None,
        stderr=None,
    )

    return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


print()
print("=" * 80)
print("Colab project setup")
print("=" * 80)


# ---------------------------------------------------------
# 1. 清理旧源码
# ---------------------------------------------------------

print()
print("[remote 1/3] 解压源码", flush=True)

if REMOTE_ROOT.exists():
    print(f"删除旧目录: {{REMOTE_ROOT}}", flush=True)
    shutil.rmtree(REMOTE_ROOT)

REMOTE_ROOT.mkdir(parents=True, exist_ok=True)

with tarfile.open(ARCHIVE, "r:gz") as tar:
    tar.extractall(REMOTE_ROOT)

print(f"源码目录: {{REMOTE_ROOT}}", flush=True)
print("✓ 解压完成", flush=True)


# ---------------------------------------------------------
# 2. 确保 uv 存在
# ---------------------------------------------------------

print()
print("[remote 2/3] 检查 uv", flush=True)

if shutil.which("uv") is None:
    print("未发现 uv，正在安装...", flush=True)

    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "uv",
    ])
else:
    run(["uv", "--version"])


# ---------------------------------------------------------
# 3. uv sync
# ---------------------------------------------------------

print()
print("[remote 3/3] 同步 Python 环境", flush=True)
print("这一步可能需要几分钟，下面会实时显示 uv 输出。", flush=True)

env = os.environ.copy()

# 避免 Python 自己缓冲输出
env["PYTHONUNBUFFERED"] = "1"

run(
    [
        "uv",
        "sync",
        "--color",
        "always",
    ],
    cwd=REMOTE_ROOT,
    env=env,
)


# 清理上传的压缩包
if ARCHIVE.exists():
    ARCHIVE.unlink()


print()
print("=" * 80)
print("✓ Colab 环境部署完成")
print()
print(f"项目目录: {{REMOTE_ROOT}}")
print()
print("接下来可以执行:")
print(f"  cd {{REMOTE_ROOT}}")
print("  uv run python ...")
print("=" * 80)
"""

    path.write_text(script, encoding="utf-8")


def deploy(session: str):
    # deploy_colab.py 在 scripts/ 下
    project_root = Path(__file__).resolve().parent.parent

    if not (project_root / "pyproject.toml").exists():
        print(
            f"错误：没有在项目根目录找到 pyproject.toml\\n"
            f"推断的项目目录：{project_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 80)
    print("Deploy project to Google Colab")
    print("=" * 80)
    print(f"Session : {session}")
    print(f"Local   : {project_root}")
    print(f"Remote  : {REMOTE_ROOT}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        archive_path = tmp / "source.tar.gz"
        remote_script = tmp / "remote_setup.py"

        create_archive(project_root, archive_path)

        upload_archive(session, archive_path)

        print()
        print("[3/4] 准备远端环境")
        create_remote_setup_script(remote_script)

        print()
        print("[4/4] 执行远端 setup")
        print("      uv sync 的输出会直接显示在下面。")

        run([
            "colab",
            "exec",
            "-s",
            session,
            "-f",
            str(remote_script),
        ])

    print()
    print("=" * 80)
    print("✓ Deployment completed")
    print()
    print("进入 Colab:")
    print(f"  colab console -s {session}")
    print()
    print("进入项目:")
    print(f"  cd {REMOTE_ROOT}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Deploy local project source to a running Google Colab session."
    )

    parser.add_argument(
        "-s",
        "--session",
        default=DEFAULT_SESSION,
        help=f"Colab session name (default: {DEFAULT_SESSION})",
    )

    args = parser.parse_args()

    deploy(args.session)


if __name__ == "__main__":
    main()
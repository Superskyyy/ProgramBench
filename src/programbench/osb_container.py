# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""OpenSandbox (k3s) grading backend mirroring ContainerEnvironment.

Selected at runtime when PROGRAMBENCH_BACKEND=opensandbox. The Docker path in
``container.py`` is untouched; ``eval.py`` imports this module lazily.

Design (SDK 0.1.x has no snapshot): one sandbox is built per instance and reused
across every test branch serially. ``commit`` registers the live sandbox in the
module-level ``_OSB`` dict keyed by a synthetic image ref; ``_new_env`` returns
the registered env when handed that ref, so all branches share the post-build
state. ``remove_image`` performs the real teardown (kill).
"""

import logging
import os
import shlex
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, TypeVar

from programbench.constants import WORKSPACE_DIR

log = logging.getLogger(__name__)

# Live sandboxes keyed by the synthetic committed-image ref handed back by commit().
_OSB: dict[str, "OpenSandboxEnvironment"] = {}

OSB_DOMAIN = os.environ.get("PROGRAMBENCH_OSB_DOMAIN", "10.10.110.23:9080")
OSB_REGISTRY = os.environ.get("PROGRAMBENCH_OSB_REGISTRY", "10.10.110.20:5000")
OSB_REGISTRY_USER = os.environ.get("PROGRAMBENCH_OSB_REGISTRY_USER", "bmc")
OSB_REGISTRY_PASS = os.environ.get("PROGRAMBENCH_OSB_REGISTRY_PASS", "bmc")
OSB_CPU = os.environ.get("PROGRAMBENCH_OSB_CPU", "4")
# 16Gi default: some test branches (e.g. fasttext's training tests) OOM-kill the
# pod at 8Gi, which restarts it and wipes /workspace mid-eval. Docker ran these
# against host memory. Override with PROGRAMBENCH_OSB_MEMORY.
OSB_MEMORY = os.environ.get("PROGRAMBENCH_OSB_MEMORY", "16Gi")
# Sandbox lifetime; must outlast compile + every branch run. Generous default.
OSB_LIFETIME_SEC = int(os.environ.get("PROGRAMBENCH_OSB_LIFETIME", "7200"))
OSB_READY_SEC = int(os.environ.get("PROGRAMBENCH_OSB_READY_TIMEOUT", "600"))
OSB_CREATE_RETRIES = int(os.environ.get("PROGRAMBENCH_OSB_CREATE_RETRIES", "4"))
# The gateway occasionally resets connections under load ("Connection reset by
# peer", read timeouts). File up/download and command runs are retried so a
# transient transport blip on one branch's results.xml read doesn't drop tests.
OSB_IO_RETRIES = int(os.environ.get("PROGRAMBENCH_OSB_IO_RETRIES", "5"))
# Fresh-sandbox-per-branch mode (mirrors local Docker). Avoids memory accumulating in one long-lived
# reused pod (which OOM-kills heavy instances like zstd/fasttext). Costs a sandbox create per branch.
OSB_FRESH_PER_BRANCH = os.environ.get("PROGRAMBENCH_OSB_FRESH_PER_BRANCH", "") not in ("", "0", "false", "no")
# Must match Evaluator._stashed_executable in eval/eval.py (candidate binary stashed outside /workspace).
_STASHED_EXECUTABLE = "/opt/programbench-stashed-executable-do-not-modify"

T = TypeVar("T")


def _with_retries(label: str, fn: Callable[[], T]) -> T:
    last: Exception | None = None
    for attempt in range(1, OSB_IO_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            last = e
            log.warning("OpenSandbox %s attempt %d/%d failed: %s", label, attempt, OSB_IO_RETRIES, e)
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"OpenSandbox {label} failed after {OSB_IO_RETRIES} attempts: {last}")
# Path to the helper that builds the gateway connection config (use_server_proxy).
OSB_PATCHES_DIR = os.environ.get(
    "PROGRAMBENCH_OSB_PATCHES_DIR",
    "/shared_workspace_mfs/yihao/miles/examples/PlannerRL/swe_blackbox_patches",
)


def _connection_config():
    import sys

    if OSB_PATCHES_DIR not in sys.path:
        sys.path.insert(0, OSB_PATCHES_DIR)
    from opensandbox_backend import _make_connection_config_sync

    return _make_connection_config_sync(
        domain=OSB_DOMAIN,
        api_key=None,
        protocol="http",
        use_server_proxy=True,
        request_timeout=timedelta(seconds=900),
    )


def map_image_to_cleanroom(image: str, instance_id: str) -> str:
    """Map a Docker image ref (``programbench/<inst>:task[_cleanroom]``) to the
    OpenSandbox registry cleanroom ref. The instance id drives the name so the
    ``__`` → ``_1776_`` rewrite matches what's pushed to the registry."""
    inst = instance_id or image.split("/", 1)[-1].split(":", 1)[0]
    return f"{OSB_REGISTRY}/programbench/{inst.replace('__', '_1776_')}:task_cleanroom"


class OpenSandboxEnvironment:
    """ContainerEnvironment-compatible backend over an OpenSandbox sandbox.

    The interface mirrors ``container.ContainerEnvironment``: ``execute``,
    ``copy_in``, ``copy_in_tar``, ``copy_out``, ``commit``, ``cleanup``. The
    Evaluator drives both backends through this surface.
    """

    def __init__(
        self,
        *,
        image: str | None = None,
        cwd: str = "/",
        executable: str = "opensandbox",
        timeout: int = 30,
        cpus: int = 10,
        env: dict[str, str] | None = None,
        run_args: list[str] | None = None,
        instance_id: str = "",
        snapshot_id: str | None = None,
    ):
        from opensandbox import SandboxSync
        from opensandbox.models.sandboxes import SandboxImageAuth, SandboxImageSpec

        self.cwd = cwd
        self.executable = executable
        self.default_timeout = timeout
        self.cpus = cpus
        self.instance_id = instance_id
        self._committed = False
        self._snapshot_id: str | None = None
        env_dict = {"PYTEST_XDIST_AUTO_NUM_WORKERS": str(cpus), **(env or {})}

        spec = None if snapshot_id else SandboxImageSpec(
            image=image,
            auth=SandboxImageAuth(username=OSB_REGISTRY_USER, password=OSB_REGISTRY_PASS),
        )
        last_exc: Exception | None = None
        for attempt in range(1, OSB_CREATE_RETRIES + 1):
            try:
                self.sb = SandboxSync.create(
                    image=spec,
                    snapshot_id=snapshot_id,
                    timeout=timedelta(seconds=OSB_LIFETIME_SEC),
                    ready_timeout=timedelta(seconds=OSB_READY_SEC),
                    resource={"cpu": OSB_CPU, "memory": OSB_MEMORY},
                    env=env_dict,
                    metadata={"name": f"spb-{(instance_id or 'inst')[:40]}", "managed-by": "spb-grade"},
                    connection_config=_connection_config(),
                )
                break
            except Exception as e:  # gateway 504s under load are transient
                last_exc = e
                log.warning("OpenSandbox create attempt %d/%d failed: %s", attempt, OSB_CREATE_RETRIES, e)
        else:
            raise RuntimeError(f"Failed to create sandbox after {OSB_CREATE_RETRIES} attempts: {last_exc}")
        self.container_id = self.sb.id

    def execute(self, command: str, *, timeout: int | None = None) -> dict[str, Any]:
        """Run a shell command in the sandbox (mirrors docker exec -w cwd bash -lc).

        Each OutputMessage is one logical line with no trailing newline, so
        streams are rejoined with ``\\n``; output is ``stdout + stderr`` to match
        the Docker backend. The return code comes from ``Execution.exit_code``
        (falls back to parsing ``error.value`` / 1 when the server omits it).
        """
        from opensandbox.exceptions.sandbox import SandboxException
        from opensandbox.models.execd import RunCommandOpts

        timeout = timeout or self.default_timeout
        wrapped = f"cd {shlex.quote(self.cwd)} && ( {command} )"
        full = f"bash -lc {shlex.quote(wrapped)}"
        opts = RunCommandOpts(timeout=timedelta(seconds=timeout))

        def _run():
            return self.sb.commands.run(full, opts=opts)

        # A command timeout (the test suite genuinely overran) is a final result,
        # not a transport blip — surface it without retrying. Other Sandbox
        # exceptions and transport errors (connection reset under gateway load)
        # are retried.
        last: Exception | None = None
        r = None
        for attempt in range(1, OSB_IO_RETRIES + 1):
            try:
                r = _run()
                break
            except SandboxException as e:
                low = str(e).lower()
                if "timeout" in low or "timed out" in low or "deadline" in low:
                    return {"output": "", "returncode": -1, "exception_info": f"Command timed out after {timeout}s"}
                last = e
            except Exception as e:  # transport-level (httpx ReadError/ConnectError, ...)
                last = e
            log.warning("OpenSandbox exec attempt %d/%d failed: %s", attempt, OSB_IO_RETRIES, last)
            time.sleep(min(2 * attempt, 10))
        if r is None:
            return {"output": "", "returncode": -1, "exception_info": str(last)}

        stdout = "\n".join(m.text for m in r.logs.stdout)
        stderr = "\n".join(m.text for m in r.logs.stderr)
        output = stdout + stderr
        rc = r.exit_code
        if rc is None:
            if r.error is not None:
                try:
                    rc = int(str(r.error.value).strip())
                except (ValueError, AttributeError):
                    rc = 1
            else:
                rc = 0
        return {"output": output, "returncode": rc, "exception_info": ""}

    def copy_in(self, local_path: Path, container_path: str) -> None:
        """Copy a local file or directory tree into the sandbox.

        Directories are tarred on the host and streamed in (preserving symlinks,
        modes) the same way ``copy_in_tar`` works; single files are uploaded
        with ``files.write_file``.
        """
        if local_path.is_dir():
            import io
            import tarfile

            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                tar.add(str(local_path), arcname=".")
            self._stream_tar_bytes(buf.getvalue(), container_path, compressed=False)
            return
        _with_retries(f"write {container_path}", lambda: self.sb.files.write_file(container_path, local_path.read_bytes()))

    def copy_in_tar(self, tar_path: Path, container_path: str) -> None:
        """Stream an on-disk tar(.gz) into the sandbox: upload to a temp path,
        then extract with the sandbox's own tar (decompresses ``.gz``)."""
        is_gz = tar_path.name.endswith((".tar.gz", ".tgz")) or tar_path.suffix == ".gz"
        self._stream_tar_bytes(tar_path.read_bytes(), container_path, compressed=is_gz)

    def _stream_tar_bytes(self, data: bytes, container_path: str, *, compressed: bool) -> None:
        tmp = f"/tmp/spb-{uuid.uuid4().hex}.tar{'.gz' if compressed else ''}"
        _with_retries(f"write {tmp}", lambda: self.sb.files.write_file(tmp, data))
        flags = "-xzf" if compressed else "-xf"
        r = self.execute(
            f"mkdir -p {shlex.quote(container_path)} && "
            f"tar -C {shlex.quote(container_path)} {flags} {shlex.quote(tmp)} && rm -f {shlex.quote(tmp)}",
            timeout=600,
        )
        if r["returncode"] != 0:
            raise RuntimeError(f"tar stream into sandbox failed: {r['output'].strip()}")

    def copy_out(self, container_path: str, host_path: Path) -> None:
        """Download a file from the sandbox to the host, raw bytes.

        The files-download API is proxied on a different port than command exec
        and intermittently 502s under load; this is the one step whose failure
        would silently drop a whole branch's tests. So: retry the files API, and
        if it keeps failing, fall back to reading the bytes through the command
        channel (``base64`` over ``commands.run``), which proxies separately.
        """
        import base64

        try:
            data = _with_retries(f"read {container_path}", lambda: self.sb.files.read_bytes(container_path))
        except Exception as e:
            log.warning("files-API read of %s failed (%s); falling back to base64 over exec", container_path, e)
            r = self.execute(f"base64 -w0 {shlex.quote(container_path)}", timeout=120)
            if r["returncode"] != 0:
                raise RuntimeError(f"base64 copy_out fallback failed: {r['output'].strip()}") from e
            data = base64.b64decode(r["output"].strip())
        Path(host_path).write_bytes(data)

    _WS_SNAPSHOT = "/opt/spb-workspace-snapshot.tar"

    def commit(self, image_ref: str) -> str:
        """Register this live sandbox under ``image_ref``. There is no image
        snapshot (SDK has none); instead we tar the post-compile ``/workspace``
        so each branch can be reset to it, mirroring Docker's fresh-container-
        per-branch semantics. The sandbox stays alive for branch reuse."""
        r = self.execute(
            f"tar -C {WORKSPACE_DIR} -cf {self._WS_SNAPSHOT} . ",
            timeout=600,
        )
        if r["returncode"] != 0:
            raise RuntimeError(f"workspace snapshot failed: {r['output'].strip()}")
        self._committed = True
        _OSB[image_ref] = self
        return image_ref

    def reset_workspace(self) -> None:
        """Restore /workspace to the post-compile snapshot taken at commit().

        Each test branch must start from the same state a fresh Docker branch
        container would; without this the reused sandbox would carry over the
        previous branch's extracted tests and run artifacts.
        """
        r = self.execute(
            f"rm -rf {WORKSPACE_DIR}/* {WORKSPACE_DIR}/.[!.]* 2>/dev/null; "
            f"tar -C {WORKSPACE_DIR} -xf {self._WS_SNAPSHOT}",
            timeout=600,
        )
        if r["returncode"] != 0:
            raise RuntimeError(f"workspace reset failed: {r['output'].strip()}")

    def cleanup(self) -> None:
        """No-op while committed (branches still need the sandbox). Otherwise
        kill the sandbox. Real teardown of a committed sandbox happens in
        ``remove_image``."""
        if self._committed:
            return
        try:
            self.sb.kill()
        except Exception as e:
            log.warning("OpenSandbox kill failed for %s: %s", self.container_id, e)


def remove_image(image_ref: str) -> None:
    """OpenSandbox teardown: kill the sandbox registered under ``image_ref``."""
    env = _OSB.pop(image_ref, None)
    if env is None:
        return
    try:
        env.sb.kill()
    except Exception as e:
        log.warning("OpenSandbox kill failed for %s: %s", env.container_id, e)

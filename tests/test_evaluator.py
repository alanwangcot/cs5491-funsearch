# CS5491: Reasoning Augmented FunSearch
# Author: 李逸洋 (72655555)
# Responsibility: Benchmark preparation and evaluation scripts

import pytest

from funsearch_project.baselines import BEST_FIT_CODE
from funsearch_project.binpacking import Instance
from funsearch_project.evaluator import evaluate_candidate, isolation_available


def test_static_hostile_code_rejected_without_execution():
    result = evaluate_candidate(
        "import os\ndef score_bins(item,remaining):\n return remaining",
        [Instance("x", 10, (6, 4))],
    )
    assert result.status == "static_error"
    assert result.bins == {}


def test_missing_isolation_fails_closed():
    result = evaluate_candidate(
        BEST_FIT_CODE,
        [Instance("x", 10, (6, 4))],
        {"image": "missing-funsearch-image:no-such-tag"},
    )
    assert result.status == "isolation_unavailable"
    assert result.fitness is None


@pytest.mark.skipif(
    not isolation_available(), reason="isolated Docker image not installed"
)
def test_isolated_worker_correct_and_no_partial_success(monkeypatch):
    monkeypatch.setenv("FAKE_CREDENTIAL", "must-not-cross-boundary")
    result = evaluate_candidate(
        BEST_FIT_CODE, [Instance("x", 10, (6, 6, 4, 4)), Instance("y", 10, (10,))]
    )
    assert result.status == "ok"
    assert result.bins == {"x": 2, "y": 1}
    assert result.fitness == 1
    bad = evaluate_candidate(
        "def score_bins(item, remaining):\n return np.log(-remaining)",
        [Instance("x", 10, (6, 4))],
    )
    assert bad.status == "runtime_error"
    assert bad.bins == {}
    assert bad.fitness is None


@pytest.mark.skipif(
    not isolation_available(), reason="isolated Docker image not installed"
)
def test_total_timeout_kills_container():
    result = evaluate_candidate(
        BEST_FIT_CODE, [Instance("x", 10, (6, 4))], {"total_seconds": 0.001}
    )
    assert result.status == "timeout"
    assert result.bins == {}


@pytest.mark.skipif(
    not isolation_available(), reason="isolated Docker image not installed"
)
def test_actual_container_security_boundary(monkeypatch):
    """Probe the real execution boundary with trusted code, independent of AST filters."""
    import json
    import subprocess
    import uuid
    from pathlib import Path

    from funsearch_project import evaluator

    monkeypatch.setenv("FAKE_BOUNDARY_CREDENTIAL", "must-not-be-inherited")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-host-key-never-forwarded")
    source = Path(evaluator.__file__).resolve().parent
    name = "funsearch-security-probe-" + uuid.uuid4().hex
    probe = """
import errno,json,os,pathlib,socket
result={'uid':os.getuid(),'fake_credential_absent':'FAKE_BOUNDARY_CREDENTIAL' not in os.environ,'api_key_absent':'DEEPSEEK_API_KEY' not in os.environ,'data_unmounted':not pathlib.Path('/app/data').exists(),'credential_file_unmounted':not pathlib.Path('/project/.env').exists()}
for label,path in [('root','/boundary-write-probe'),('source','/app/funsearch_project/boundary-write-probe')]:
 try:
  pathlib.Path(path).write_text('forbidden')
  result[label+'_readonly']=False
 except OSError as exc:
  result[label+'_readonly']=exc.errno in (errno.EROFS,errno.EACCES)
pathlib.Path('/tmp/permitted-probe').write_text('permitted')
result['temporary_write_allowed']=pathlib.Path('/tmp/permitted-probe').read_text()=='permitted'
sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM);sock.settimeout(.5)
result['network_unreachable']=sock.connect_ex(('8.8.8.8',53)) in (errno.ENETUNREACH,errno.EHOSTUNREACH,errno.EPERM)
sock.close()
status=pathlib.Path('/proc/self/status').read_text().splitlines()
fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in status if ':' in line}
result['no_new_privileges']=fields['NoNewPrivs']=='1'
result['effective_capabilities_dropped']=int(fields['CapEff'],16)==0
result['memory_limit']=pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip()
result['pid_limit']=pathlib.Path('/sys/fs/cgroup/pids.max').read_text().strip()
print(json.dumps(result))
"""
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "32",
        "--memory",
        "512m",
        "--memory-swap",
        "512m",
        "--cpus",
        "1",
        "--user",
        "65534:65534",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",
        "--mount",
        f"type=bind,source={source},target=/app/funsearch_project,readonly",
        "--entrypoint",
        "python",
        evaluator.IMAGE,
        "-I",
        "-c",
        probe,
    ]
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, timeout=20, check=True
        )
        result = json.loads(completed.stdout)
        assert result["uid"] == 65534
        for field in (
            "fake_credential_absent",
            "api_key_absent",
            "data_unmounted",
            "credential_file_unmounted",
            "root_readonly",
            "source_readonly",
            "temporary_write_allowed",
            "network_unreachable",
            "no_new_privileges",
            "effective_capabilities_dropped",
        ):
            assert result[field] is True, field
        assert result["memory_limit"] == str(512 * 1024 * 1024)
        assert result["pid_limit"] == "32"
    finally:
        subprocess.run(
            ["docker", "rm", "--force", name], capture_output=True, timeout=10
        )

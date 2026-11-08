"""Run lifecycle states shared by runners, suites and the dashboard.

queued       planned in a suite, not started yet
running      process alive and reporting a heartbeat
stalled      process alive but no heartbeat for STALL_SECONDS
completed    finished normally
stopped      stopped on purpose (SIGINT / SIGTERM / Ctrl-C); partial results were saved
failed       raised an exception
interrupted  status says running but the process is gone (killed -9, crash, reboot)
"""
import json
import os
from pathlib import Path
import signal
import socket
import time

STATES = ('queued', 'running', 'stalled', 'completed', 'stopped', 'failed', 'interrupted')
FINAL = ('completed', 'stopped', 'failed', 'interrupted')
STALL_SECONDS = 600


class RunStopped(Exception):
    """Raised inside a run when it receives SIGINT or SIGTERM."""


def read(path):
    try:
        return json.loads((Path(path)/'status.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def write(path, status, **fields):
    """Atomically update status.json, keeping identity fields (pid, host, started) from earlier writes."""
    if status not in STATES:
        raise ValueError(f'Unknown run status: {status}')
    path = Path(path)
    record = {k: v for k, v in read(path).items() if k in ('pid', 'host', 'started', 'day', 'epochs')}
    record.update(fields, status=status, updated=time.time())
    temporary = path/'status.json.tmp'
    temporary.write_text(json.dumps(record), encoding='utf-8')
    temporary.replace(path/'status.json')
    return record


def start(path, **fields):
    return write(path, 'running', pid=os.getpid(), host=socket.gethostname(), started=time.time(), **fields)


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError, OSError):
        return False
    return True


def effective(record, now=None):
    """Status as observed now: detects dead (interrupted) and silent (stalled) runs."""
    record = dict(record or {})
    status = record.get('status', 'unknown')
    if status == 'running':
        same_host = record.get('host') in (None, socket.gethostname())
        if same_host and record.get('pid') and not pid_alive(record['pid']):
            status = 'interrupted'
        elif now is not None or record.get('updated'):
            age = (now or time.time()) - record.get('updated', record.get('started', 0))
            if age > STALL_SECONDS:
                status = 'stalled'
    record['raw_status'] = record.get('status')
    record['status'] = status
    return record


def install_stop_handlers():
    """Turn SIGINT/SIGTERM into RunStopped so the runner can save partial results and record 'stopped'."""
    def handler(signum, frame):
        raise RunStopped(signal.Signals(signum).name)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handler)

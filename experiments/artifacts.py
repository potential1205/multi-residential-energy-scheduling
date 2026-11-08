"""Immutable per-run directories and content-addressed artifact manifests."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import uuid

DIRECTORIES=('source','environment','metrics','checkpoints','evaluation','history','logs','figures')

def default_run_path(root,algorithm,source,scenario,seed):
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return Path(root)/'results'/'runs'/f'{timestamp}__{algorithm.lower()}-{source}__s{scenario}__seed{seed}__{uuid.uuid4().hex[:6]}'

def prepare(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    for directory in DIRECTORIES:(path/directory).mkdir()
    return path

def manifest(path,status,config):
    path=Path(path);files=[]
    for file in sorted(path.rglob('*')):
        if file.is_file() and file.name!='manifest.json':
            files.append(dict(path=file.relative_to(path).as_posix(),bytes=file.stat().st_size,
                              sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    record=dict(schema_version=1,status=status,run_id=path.name,created_at=datetime.now(timezone.utc).isoformat(),
                config_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest(),
                source_sha256=config.get('executed_source_sha256'),files=files)
    (path/'manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')

def data_manifest(root):
    return [dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,
                 sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted((Path(root)/'data').rglob('*.npy'))]

"""Bounded production delivery on robot2. Private state never leaves this host."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import time
import urllib.request
from zipfile import ZipFile

PROD = Path('/home/jack/smartfolio')
ROOT = Path('/home/jack/smartfolio-releases/ml-reliability-20261002')
OLD = 'sha256:0c724b8b8ff87d73885bf0e109276c7b1da2b211f3c9c497b33f9bcda544f2aa'
START = '2026-09-28T09:01:59.068663028Z'
BASE = '806338b522e89647c3f49260e0f418dbea1da761'
NAME = 'smartfolio-api'
BACKUP = 'smartfolio-api-before-ml-reliability-20261002'
CHECK = 'smartfolio-ml-production-check-20261002'
IMAGE = 'smartfolio-prod:ml-reliability-20261002'
BASE_TAG = 'smartfolio-prod-base:806338-20261002'


def run(args):
    return subprocess.check_output(args, stderr=subprocess.PIPE)


def inspect(name):
    return json.loads(run(['docker', 'inspect', name]))[0]


def sha(content):
    return hashlib.sha256(content).hexdigest()


def write_private(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as out:
        json.dump(value, out)


def emit(value, filename=None):
    if filename:
        (ROOT / filename).write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)


def invariant():
    assert run(['hostname']).decode().strip() == 'robot2', 'Unexpected host'
    assert ROOT.resolve().is_relative_to(Path('/home/jack/smartfolio-releases').resolve())
    current = inspect(NAME)
    assert current['Image'] == OLD and current['State']['StartedAt'] == START and current['State']['Running'], 'Production baseline changed'
    assert run(['git', '-C', str(PROD), 'rev-parse', 'HEAD']).decode().strip() == BASE, 'Production checkout changed'
    return current


def verified_package(path, expected):
    assert sha(path.read_bytes()) == expected, 'Package digest mismatch'
    with ZipFile(path) as z:
        manifest = json.loads(z.read('manifest.json'))
        assert manifest['base_commit'] == BASE and manifest['private_data_included'] is False
        for row in manifest['entries']:
            name = row['path']
            assert not any(x in name for x in ('data/users/', '.env', 'private/')), 'Private entry rejected'
            assert sha(z.read(name)) == row['sha256'], 'Entry digest mismatch'
        return manifest


def build(args):
    invariant()
    assert shutil.disk_usage(PROD).free > 15 * 1024**3, 'Insufficient free disk'
    ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = verified_package(args.package, args.sha)
    context = ROOT / 'context'
    context.mkdir(exist_ok=True)
    with ZipFile(args.package) as z:
        for row in manifest['entries']:
            if not row['path'].startswith(('code/', 'public/models/validated_risk/', 'public/data/ml_verified/')):
                continue
            target = (context / row['path']).resolve()
            assert target.is_relative_to(context.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(row['path']))
    (context / 'Dockerfile').write_text('''ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
WORKDIR /app
COPY code/ /app/
RUN pip install --no-cache-dir "exchange-calendars==4.13.2" && pip check
COPY public/models/validated_risk/ /app/models/validated_risk/
ENV PYTHONDONTWRITEBYTECODE=1 ML_AUTO_TRAIN=0
LABEL smartfolio.release="ml-reliability-20261002"
''')
    subprocess.run(['docker', 'tag', OLD, BASE_TAG], check=True)
    with (ROOT / 'build.log').open('wb') as log:
        p = subprocess.run(['docker', 'build', '--build-arg', 'BASE_IMAGE='+BASE_TAG,
                            '--label', 'org.opencontainers.image.revision='+args.commit,
                            '--label', 'smartfolio.package.sha256='+args.sha,
                            '-t', IMAGE, str(context)], stdout=log, stderr=subprocess.STDOUT)
    assert p.returncode == 0, 'Build failed; inspect public build.log on robot2'
    code = '''import pathlib,json
from services.ml.reliability import code_version
p=pathlib.Path('/app')
assert not (p/'data/users/jack').exists() and not (p/'.env').exists()
assert not (p/'preview_app.py').exists() and not (p/'preview_guard.py').exists()
assert not (p/'static/preview-banner.js').exists()
registry=json.loads((p/'models/validated_risk/registry.json').read_text())
assert len(registry)==62
assert all(json.loads((p/'models/validated_risk'/k).read_text())['code_version']==code_version() for k in registry)
print(json.dumps({'private_account_not_baked':True,'preview_runtime_absent':True,'artifacts':len(registry),'code_fingerprints_match':True}))
'''
    subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'python', IMAGE, '-B', '-c', code], check=True)
    invariant()
    record = dict(package_sha256=args.sha, commit=args.commit, image=inspect(IMAGE)['Id'], entries=len(manifest['entries']), production_unchanged=True)
    emit(record, 'build-result.json')


class DockerHTTP(http.client.HTTPConnection):
    def __init__(self):
        super().__init__('localhost')

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect('/var/run/docker.sock')


def api(method, path, payload=None):
    conn = DockerHTTP()
    conn.request(method, path, json.dumps(payload).encode() if payload is not None else None, {'Content-Type': 'application/json'})
    response = conn.getresponse()
    raw = response.read()
    conn.close()
    assert response.status < 300, 'Docker operation failed: '+str(response.status)
    return json.loads(raw) if raw else {}


def config_from(old):
    keys = ('User', 'Env', 'Cmd', 'Healthcheck', 'WorkingDir', 'Entrypoint', 'StopSignal', 'StopTimeout', 'Shell', 'Labels', 'Tty', 'OpenStdin', 'StdinOnce', 'ExposedPorts', 'Volumes')
    config = {k: v for k, v in old['Config'].items() if k in keys}
    config['Image'] = IMAGE
    config['Env'] = [x for x in config['Env'] if not x.startswith(('ML_AUTO_TRAIN=', 'ML_PORTFOLIO_SNAPSHOT='))] + ['ML_AUTO_TRAIN=0']
    config['Labels'] = {**(config.get('Labels') or {}), **inspect(IMAGE)['Config']['Labels']}
    config['HostConfig'] = json.loads(json.dumps(old['HostConfig']))
    config['NetworkingConfig'] = {'EndpointsConfig': {n: {'Aliases': [NAME]} for n in old['NetworkSettings']['Networks']}}
    return config


def ready(port):
    try:
        host = '127.0.0.1' if port == 8084 else '192.168.1.200'
        return urllib.request.urlopen('http://'+host+':'+str(port)+'/healthz', timeout=3).status == 200
    except Exception:
        return False


def wait_ready(name, port):
    deadline = time.monotonic()+150
    while time.monotonic() < deadline:
        if ready(port):
            return
        assert inspect(name)['State']['Running'], 'Candidate container stopped'
        time.sleep(1)
    raise RuntimeError('Readiness timeout')


def preflight(args):
    old = invariant()
    assert args.commit == inspect(IMAGE)['Config']['Labels']['org.opencontainers.image.revision']
    assert not (ROOT / 'preflight').exists(), 'Preflight already exists; preserve its results'
    folder = ROOT / 'preflight'
    folder.mkdir(mode=0o700)
    assert not any(p.is_symlink() for p in (PROD / 'data').rglob('*')), 'Reject symlinked data for write-path validation'
    shutil.copytree(PROD / 'data', folder / 'data')
    (folder / 'logs').mkdir()
    (folder / 'cache').mkdir()
    assert not (folder / 'data/ml_verified').exists(), 'Public dataset target already exists'
    shutil.copytree(ROOT / 'context/public/data/ml_verified', folder / 'data/ml_verified')
    config = config_from(old)
    env = dict(x.split('=', 1) for x in config['Env'] if '=' in x)
    env.update(RUN_SCHEDULER='0', ML_AUTO_TRAIN='0', ML_INFERENCE_JOURNAL='0',
               JWT_SECRET_KEY=secrets.token_urlsafe(48), REDIS_URL='redis://smartfolio-ml-preview-redis:6379/3',
               ALLOWED_HOSTS='192.168.1.200,localhost,127.0.0.1,testserver', API_BASE_URL='http://127.0.0.1:8084')
    config['Env'] = [k+'='+v for k, v in env.items()]
    host = config['HostConfig']
    host['Binds'] = [str(folder/'data')+':/app/data:rw', str(folder/'logs')+':/app/logs:rw', str(folder/'cache')+':/app/cache:rw']
    host['PortBindings'] = {'8080/tcp': [{'HostIp': '127.0.0.1', 'HostPort': '8084'}]}
    host['RestartPolicy'] = {'Name': 'no', 'MaximumRetryCount': 0}
    host['Memory'] = 3 * 1024**3
    host['NanoCpus'] = 2_000_000_000
    config['Labels'] = {'smartfolio.preflight': 'ml-reliability-20261002'}
    config['NetworkingConfig'] = {'EndpointsConfig': {'smartfolio-ml-preview-network': {'Aliases': [CHECK]}}}
    created = api('POST', '/containers/create?name='+CHECK, config)
    api('POST', '/containers/'+created['Id']+'/start')
    wait_ready(CHECK, 8084)
    invariant()
    emit({'preflight_ready': True, 'private_copy_stays_on_robot2': True, 'scheduler_disabled': True, 'auto_training_disabled': True, 'production_unchanged': True})


def stop_preflight(args):
    d = inspect(CHECK)
    assert d['Config'].get('Labels', {}).get('smartfolio.preflight') == 'ml-reliability-20261002'
    subprocess.run(['docker', 'stop', CHECK], check=True, stdout=subprocess.DEVNULL)
    emit({'preflight_stopped': True})


def switch(args):
    old = invariant()
    build_record = json.loads((ROOT / 'build-result.json').read_text())
    assert args.commit == build_record['commit'] and inspect(IMAGE)['Id'] == build_record['image']
    assert json.loads((ROOT / 'preflight-validation.json').read_text())['all_passed'] is True, 'Preflight validation required'
    names = run(['docker', 'ps', '-a', '--format', '{{.Names}}']).decode().splitlines()
    assert BACKUP not in names, 'Rollback container already exists'
    private = ROOT / 'private-backup'
    private.mkdir(mode=0o700)
    write_private(private / 'container.json', old)
    config = config_from(old)
    subprocess.run(['docker', 'stop', NAME], check=True, stdout=subprocess.DEVNULL)
    renamed = False
    try:
        shutil.copytree(PROD / 'data', private / 'data', symlinks=True)
        assert not (PROD / 'data/ml_verified').exists(), 'Public dataset target already exists'
        temp = PROD / 'data/ml_verified.delivery-20261002'
        assert not temp.exists()
        shutil.copytree(ROOT / 'context/public/data/ml_verified', temp)
        temp.rename(PROD / 'data/ml_verified')
        subprocess.run(['docker', 'rename', NAME, BACKUP], check=True)
        renamed = True
        created = api('POST', '/containers/create?name='+NAME, config)
        api('POST', '/containers/'+created['Id']+'/start')
        wait_ready(NAME, 8080)
        current = inspect(NAME)
        assert current['HostConfig']['PortBindings'] == old['HostConfig']['PortBindings']
        assert {(m['Source'], m['Destination'], m['RW']) for m in current['Mounts']} == {(m['Source'], m['Destination'], m['RW']) for m in old['Mounts']}
        a = dict(x.split('=', 1) for x in old['Config']['Env'] if '=' in x)
        b = dict(x.split('=', 1) for x in current['Config']['Env'] if '=' in x)
        assert {k: v for k, v in a.items() if k not in ('ML_AUTO_TRAIN', 'ML_PORTFOLIO_SNAPSHOT')} == {k: v for k, v in b.items() if k not in ('ML_AUTO_TRAIN', 'ML_PORTFOLIO_SNAPSHOT')}
        assert b['ML_AUTO_TRAIN'] == '0' and b.get('RUN_SCHEDULER') == a.get('RUN_SCHEDULER')
        emit({'status': 'production_ready', 'image': current['Image'], 'commit': args.commit, 'started': current['State']['StartedAt'], 'rollback_container': BACKUP, 'private_backup_local_only': True, 'mounts_preserved': True, 'scheduler_preserved': True, 'auto_training_disabled': True}, 'deployment-result.json')
    except Exception:
        if renamed:
            rollback(args)
        else:
            subprocess.run(['docker', 'start', NAME], check=True, stdout=subprocess.DEVNULL)
        raise


def rollback(args):
    backup = inspect(BACKUP)
    assert backup['Image'] == OLD and not backup['State']['Running']
    names = run(['docker', 'ps', '-a', '--format', '{{.Names}}']).decode().splitlines()
    if NAME in names:
        current = inspect(NAME)
        assert current['Config'].get('Labels', {}).get('smartfolio.release') == 'ml-reliability-20261002'
        subprocess.run(['docker', 'stop', NAME], check=True, stdout=subprocess.DEVNULL)
        failed = NAME+'-failed-ml-reliability-20261002'
        assert failed not in names
        subprocess.run(['docker', 'rename', NAME, failed], check=True)
    subprocess.run(['docker', 'rename', BACKUP, NAME], check=True)
    subprocess.run(['docker', 'start', NAME], check=True, stdout=subprocess.DEVNULL)
    wait_ready(NAME, 8080)
    emit({'status': 'rolled_back', 'private_data_not_overwritten': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'preflight', 'stop-preflight', 'switch', 'rollback'])
    parser.add_argument('--package', type=Path)
    parser.add_argument('--sha')
    parser.add_argument('--commit')
    arguments = parser.parse_args()
    {'build': build, 'preflight': preflight, 'stop-preflight': stop_preflight, 'switch': switch, 'rollback': rollback}[arguments.action](arguments)

"""Private operator setup: real temporary files with a controlled hash utility."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("demo_deployment_configure_tests", ROOT / "deploy/demo/configure.py")
configure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configure)

PASSWORD = "controlled-demo-pass-123"
KEY = "controlled-provider-placeholder"
HASH = b"$2y$10$" + b"A" * 53


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*args, **kwargs): raise AssertionError("configuration tests cannot contact a service")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


@pytest.fixture
def hash_tool(monkeypatch):
    seen=[]
    monkeypatch.setattr(configure.shutil,"which",lambda name:"/usr/bin/htpasswd")
    def run(args, **kwargs):
        seen.append((list(args),dict(kwargs)))
        assert PASSWORD not in " ".join(args)
        assert "-b" not in args
        assert kwargs['input']==(PASSWORD+'\n').encode()
        assert kwargs['timeout']==20 and kwargs['check'] is True
        return SimpleNamespace(stdout=args[-1].encode()+b":"+HASH+b"\n")
    monkeypatch.setattr(configure.subprocess,"run",run)
    return seen


@pytest.fixture
def private_dir(tmp_path):
    return tmp_path.resolve()/"new-private"


def initialize(path):
    values=iter([PASSWORD,PASSWORD,KEY])
    configure.initialize(path,"https://demo.example.test","customer-a",
        "https://dashscope.aliyuncs.com/apps/anthropic",100,read_secret=lambda prompt:next(values))


def test_new_setup_is_private_and_split_credentials_are_consistent(private_dir,hash_tool):
    initialize(private_dir)
    assert private_dir.stat().st_mode & 0o777 == 0o700
    assert {p.name for p in private_dir.iterdir()}==configure.NAMES
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in private_dir.iterdir())
    api=json.loads((private_dir/'api.json').read_text())
    gateway=json.loads((private_dir/'gateway.json').read_text())
    assert set(gateway)=={'version','public_origin','gateway_bearer'}
    assert all(gateway[k]==api[k] for k in gateway)
    assert api['api_key']==KEY and api['model_limit']==100
    assert KEY not in (private_dir/'gateway.json').read_text()
    assert PASSWORD not in ''.join(p.read_text() for p in private_dir.iterdir())
    assert len(hash_tool)==1
    assert configure.verify(private_dir)=={'result':'private_configuration_verified','users':1,'model_limit':100,'model_requests':0}


def test_existing_directory_is_never_reinitialized_or_prompted(private_dir,hash_tool):
    initialize(private_dir)
    originals={p.name:p.read_bytes() for p in private_dir.iterdir()}
    def no_prompt(*args):raise AssertionError('must stop before any secret prompt')
    with pytest.raises(configure.ConfigurationError,match='existing_configuration'):
        configure.initialize(private_dir,"https://demo.example.test","customer-a",
            "https://dashscope.aliyuncs.com/apps/anthropic",100,read_secret=no_prompt)
    assert {p.name:p.read_bytes() for p in private_dir.iterdir()}==originals
    assert len(hash_tool)==1


def test_hash_utility_receives_password_only_on_stdin(hash_tool):
    assert configure.password_hash('customer-a',PASSWORD)==b'customer-a:'+HASH+b'\n'
    argv,options=hash_tool[0]
    assert argv==['/usr/bin/htpasswd','-nBi','-C','10','customer-a']
    assert options['stdout']==subprocess.PIPE and options['stderr']==subprocess.PIPE


@pytest.mark.parametrize('username',['a:b','a\nb','bad user','a/../b','a'*65,''])
def test_invalid_username_stops_before_process(username,monkeypatch):
    monkeypatch.setattr(configure.subprocess,'run',lambda *a,**k:pytest.fail('must not spawn'))
    with pytest.raises(configure.ConfigurationError):configure.password_hash(username,PASSWORD)


@pytest.mark.parametrize('password',['short','p'*65,'a'*11+'\n'])
def test_invalid_password_stops_before_process(password,monkeypatch):
    monkeypatch.setattr(configure.subprocess,'run',lambda *a,**k:pytest.fail('must not spawn'))
    with pytest.raises(configure.ConfigurationError):configure.password_hash('customer-a',password)


@pytest.mark.parametrize('failure',['missing','timeout','bad_hash','wrong_user','process_error'])
def test_hash_utility_failures_are_safe(failure,monkeypatch):
    monkeypatch.setattr(configure.shutil,'which',lambda _:None if failure=='missing' else '/usr/bin/htpasswd')
    def run(args,**kwargs):
        if failure=='timeout':raise subprocess.TimeoutExpired(args,20,output=b'private-password-fragment')
        if failure=='process_error':raise subprocess.CalledProcessError(1,args,stderr=b'private-password-fragment')
        return SimpleNamespace(stdout=(b'other:'+HASH+b'\n') if failure=='wrong_user' else b'not-bcrypt')
    monkeypatch.setattr(configure.subprocess,'run',run)
    with pytest.raises(configure.ConfigurationError) as caught:configure.password_hash('customer-a',PASSWORD)
    assert 'private-password-fragment' not in str(caught.value)


def test_password_mismatch_leaves_no_directory(private_dir,hash_tool):
    values=iter([PASSWORD,'different-password'])
    with pytest.raises(configure.ConfigurationError,match='confirmation_mismatch'):
        configure.initialize(private_dir,"https://demo.example.test","customer-a",
            "https://dashscope.aliyuncs.com/apps/anthropic",100,read_secret=lambda _:next(values))
    assert not private_dir.exists() and not hash_tool


def test_partial_write_failure_preserves_files_and_blocks_reinit(private_dir,hash_tool,monkeypatch):
    original=configure.write_new
    def write(path,raw):
        if path.name=='gateway.json':raise OSError('controlled disk fault')
        return original(path,raw)
    monkeypatch.setattr(configure,'write_new',write)
    with pytest.raises(OSError):initialize(private_dir)
    assert (private_dir/'api.json').is_file() and not (private_dir/'gateway.json').exists()
    old=(private_dir/'api.json').read_bytes()
    with pytest.raises(configure.ConfigurationError,match='existing_configuration'):initialize(private_dir)
    assert (private_dir/'api.json').read_bytes()==old


@pytest.mark.parametrize('damage',['unknown_file','gateway_bearer','duplicate_user','bad_hash','root_secret'])
def test_manifest_credential_mismatch_and_bad_password_files_reject(private_dir,hash_tool,damage):
    initialize(private_dir)
    if damage=='unknown_file':(private_dir/'unexpected').touch()
    elif damage=='gateway_bearer':
        p=private_dir/'gateway.json';d=json.loads(p.read_text());d['gateway_bearer']='f'*64;p.write_text(json.dumps(d))
    elif damage=='duplicate_user':
        p=private_dir/'htpasswd';p.write_bytes(p.read_bytes()*2)
    elif damage=='bad_hash':(private_dir/'htpasswd').write_bytes(b'customer-a:plaintext\n')
    else:(private_dir/'mysql-root-password').write_bytes(b'short')
    with pytest.raises(configure.ConfigurationError):configure.verify(private_dir)


def test_checkout_relative_symlink_and_permissive_directories_reject(private_dir,hash_tool,tmp_path):
    with pytest.raises(configure.ConfigurationError):configure.directory(ROOT/'private-secret-tests',exists=False)
    with pytest.raises(configure.ConfigurationError):configure.directory('relative-private',exists=False)
    initialize(private_dir)
    link=tmp_path.resolve()/'private-link';link.symlink_to(private_dir,target_is_directory=True)
    with pytest.raises(configure.ConfigurationError):configure.directory(link,exists=True)
    private_dir.chmod(0o755)
    with pytest.raises(configure.ConfigurationError):configure.verify(private_dir)


def test_existing_secret_files_cannot_be_overwritten_via_link_or_hardlink(private_dir,hash_tool,tmp_path):
    initialize(private_dir)
    p=private_dir/'api.json';old=p.read_bytes()
    with pytest.raises(FileExistsError):configure.write_new(p,b'changed')
    symbolic=tmp_path.resolve()/'sym';symbolic.symlink_to(p)
    with pytest.raises(FileExistsError):configure.write_new(symbolic,b'changed')
    hard=tmp_path.resolve()/'hard';os.link(p,hard)
    with pytest.raises(FileExistsError):configure.write_new(hard,b'changed')
    with pytest.raises(configure.ConfigurationError):configure.verify(private_dir)
    assert p.read_bytes()==old


def test_special_directory_permission_bits_reject(private_dir,hash_tool):
    initialize(private_dir)
    private_dir.chmod(0o2700)
    with pytest.raises(configure.ConfigurationError):configure.verify(private_dir)


def test_check_cli_prints_only_safe_summary(private_dir,hash_tool,capsys):
    initialize(private_dir)
    assert configure.main(['check','--directory',str(private_dir)])==0
    out=capsys.readouterr().out
    assert json.loads(out)['model_requests']==0
    assert PASSWORD not in out and KEY not in out
    assert json.loads((private_dir/'api.json').read_text())['gateway_bearer'] not in out

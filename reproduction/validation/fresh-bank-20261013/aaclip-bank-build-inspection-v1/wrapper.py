import contextlib, hashlib, io, json, os, runpy, sys, traceback
from pathlib import Path
from datetime import datetime, timezone

base=Path('/home/jinyoungkim/ted-reproduce-20261009')
repo=base/'github-aaclip-bank-build-20261013-v1'
workspace=base/'aaclip-cold-runs-20261013-v2/aaclip-vitbplus-mvtec2btad-legacy-seed0'
sys.path.insert(0,str(repo)); os.chdir(repo)
from reproduction.runtime import DATA_ROOTS
from reproduction.aaclip_bank_build import collect
root=repo/'reproduction'
hashes={x['sha256'] for x in json.loads((root/'host-source-banks.json').read_bytes())['artifacts']}
denied={Path(p).resolve() for p in DATA_ROOTS['btad']}
denied.add((workspace/'source/neurips2026/results/bank_cache').resolve())
sentinel=workspace/'unavailable-historical-bank.pt'
events=[]
def audit(event,values):
    if event in {'socket.connect','socket.getaddrinfo'}:
        events.append(event); raise PermissionError('Network denied')
    if event=='open' and isinstance(values[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(values[0])).resolve()
        if p.name in hashes or p==sentinel.resolve() or any(p==t or t in p.parents for t in denied):
            events.append(str(p)); raise PermissionError('Target and historical/cache banks denied')
sys.addaudithook(audit)
proof=dict(status='running',started=datetime.now(timezone.utc).isoformat(),
           bank_collection_performed=False,GPU_submitted=False,target_evaluation_performed=False,
           prototype_overlay=True,workspace=str(workspace))
try:
    for p in [sentinel,Path(DATA_ROOTS['btad'][0])/'denial-selfcheck',
              workspace/'source/neurips2026/results/bank_cache/denial-selfcheck']:
        try: p.open('rb')
        except PermissionError: continue
        raise AssertionError('Denial selfcheck failed')
    proof['denial_selfchecks']=len(events); events.clear()
    sys.argv=['reproduction.aaclip_bank_build',str(workspace),'--inspect']
    buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer):
        runpy.run_module('reproduction.aaclip_bank_build',run_name='__main__')
    inspection=json.loads(buffer.getvalue())
    proof['inspection']=inspection
    proof['actual_CPU_CLI_inspection']=True
    destination=base/'fresh-banks/aaclip-unallocated-denial-20261013-v1'
    assert not destination.exists()
    try: collect(root,workspace,destination)
    except RuntimeError as error:
        proof['unallocated_collection_rejection']=str(error)
    else: raise AssertionError('Unallocated collection accepted')
    assert not destination.exists()
    proof.update(unallocated_collection_rejected=True,destination_created=False,
                 blocked_reads_after_selfcheck=list(events),status='completed')
    assert not events
    digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    proof.update(code_sha256=digest(repo/'reproduction/aaclip_bank_build.py'),
                 plan_sha256=digest(workspace/'run.json'),helper_sha256=digest(Path(__file__)))
    print(json.dumps(proof,indent=2))
except Exception as error:
    proof.update(status='failed',error=f'{type(error).__name__}: {error}')
    traceback.print_exc(); raise
finally:
    proof['finished']=datetime.now(timezone.utc).isoformat()
    (base/'reports/aaclip-bank-build-inspection-proof-20261013-v1.json').write_text(json.dumps(proof,indent=2)+'\n')

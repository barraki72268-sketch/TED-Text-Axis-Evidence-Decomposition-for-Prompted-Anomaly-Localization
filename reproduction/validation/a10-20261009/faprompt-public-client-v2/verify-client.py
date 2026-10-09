from pathlib import Path
import sys,json,subprocess
from reproduction.faprompt_release import prepare_release
key=sys.argv[1];r=Path('..').resolve();dest=r/'serving-bundles'/('faprompt-'+key+'-anonymous-20261009-v1');cache=r/'faprompt-public-client-cache-20261009-v1'
report=prepare_release(key,dest,cache)
with (r/'reports'/('faprompt-'+key+'-anonymous-acquisition-20261009-v1.json')).open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report),flush=True)
subprocess.run([sys.executable,'-m','reproduction.faprompt_image_parity',str(dest),str(r/'data/btad-images'),'--output',str(r/'reports'/('faprompt-'+key+'-anonymous-image-parity-20261009-v1.json'))],check=True)

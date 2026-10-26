"""Independent fresh fit in a temporary directory; preserve original artifacts."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='4'
os.environ['MPLCONFIGDIR']='/tmp/outage-mpl-cache'
from pathlib import Path
import sys,tempfile,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
import scripts.run_experiments as runner
with tempfile.TemporaryDirectory(prefix='outage-reproduction-') as tmp:
 new=Path(tmp)
 for d in ['results','models','docs']:(new/d).mkdir()
 (new/'data').symlink_to(ROOT/'data',target_is_directory=True)
 (new/'docs/PROTOCOL.md').symlink_to(ROOT/'docs/PROTOCOL.md')
 runner.ROOT=new
 with threadpool_limits(limits=4):runner.run('aps',11)
 original=np.load(ROOT/'results/aps_11/predictions.npz');repeated=np.load(new/'results/aps_11/predictions.npz')
 differences={key:float(np.max(np.abs(original[key].astype(float)-repeated[key].astype(float)))) for key in original.files}
 report={'dataset':'aps','seed':11,'arrays_compared':len(differences),'max_absolute_difference':max(differences.values()),'all_arrays_match_at_1e-12':all(v<=1e-12 for v in differences.values()),'details':differences}
 (ROOT/'results/reproduction_check.json').write_text(json.dumps(report,indent=2))
 if not report['all_arrays_match_at_1e-12']:raise SystemExit('Reproduction failed')
 print(json.dumps(report,indent=2))

"""Create an ignored local configured workspace; never connect or submit."""
from pathlib import Path
import argparse,json,re,shutil

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paper-account',required=True)
    p.add_argument('--destination',default='.local/paper_workspace')
    a=p.parse_args()
    if re.fullmatch(r'[A-Za-z0-9]{4,32}',a.paper_account) is None:
        raise ValueError('explicit account identifier required')
    root=Path(__file__).resolve().parents[1];dst=Path(a.destination).resolve()
    if dst.exists() or dst==root:raise ValueError('fresh separate local destination required')
    dst.mkdir(parents=True)
    for folder in ['training','trader_runtime','trading_desk','pa_tws']:
        shutil.copytree(root/folder,dst/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc','tokenizer','runs','audit'))
    for f in dst.rglob('*.py'):
        t=f.read_text(encoding='utf-8')
        if 'PAPER_ACCOUNT' in t:f.write_text(t.replace('PAPER_ACCOUNT',a.paper_account),encoding='utf-8')
    print(json.dumps({'configured_local_workspace':str(dst),'provider_calls':0,'broker_calls':0}))
if __name__=='__main__':main()

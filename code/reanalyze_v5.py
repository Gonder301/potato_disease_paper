"""Portable, CPU-only reanalysis of a completed extracted RESULTS_closure directory."""
import argparse,json,itertools,shutil
from pathlib import Path
import numpy as np,pandas as pd
import runtime_v5 as runtime
import engine_v5 as engine

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();source=args.results.resolve();out=args.output.resolve()
    if out.exists():raise ValueError('Use a new output directory; existing files are not overwritten.')
    if out.is_relative_to(source):raise ValueError('Output must be outside the results directory.')
    state,plans,count=runtime.validate(source)
    if count!=240:raise ValueError(f'Expected 240 verified runs, found {count}')
    print('PASS: 240 runs, frozen protocol, labels, prediction hashes, probabilities and metrics',flush=True)
    out.mkdir(parents=True)
    target=out/'recalculated';shutil.copytree(source,target)
    cfg={**engine.DEFAULT,**{k:v for k,v in state['scientific'].items() if k in engine.DEFAULT}}
    class Log:
        def log(self,text):print(text,flush=True)
    engine.summarize(target,cfg,runtime.SIGNATURE,Log())
    keys=['domain','model','metric']
    original=pd.read_csv(source/'stats.csv').sort_values(keys).reset_index(drop=True)
    fresh=pd.read_csv(target/'stats.csv').sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(original,fresh,check_exact=False,atol=1e-12,rtol=1e-12)
    metrics=pd.read_csv(target/'metrics_by_run.csv');rows=[]
    signs=np.array(list(itertools.product([-1,1],repeat=15)))
    for domain,g in metrics.groupby('domain'):
        for metric in ['accuracy','balanced_accuracy','macro_f1']:
            p=g.pivot(index=['seed','model'],columns='strategy',values=metric)
            gap=(p.random-p.group).unstack('model')
            c=(gap.vit_base_patch16_224-gap[[m for m in gap if m!='vit_base_patch16_224']].mean(axis=1)).to_numpy()
            indices=np.random.default_rng(321).integers(0,15,size=(5000,15))
            lo,hi=np.percentile(c[indices].mean(1),[2.5,97.5])
            rows.append(dict(domain=domain,metric=metric,contrast_pp=100*c.mean(),ci_low_pp=100*lo,ci_high_pp=100*hi,p_signflip=float((abs(signs@c/15)>=abs(c.mean())-1e-14).mean())))
    interaction=pd.DataFrame(rows);interaction['q_bh_6']=engine.benjamini_hochberg(interaction.p_signflip)
    interaction.to_csv(out/'interaccion_exploratoria.csv',index=False)
    (out/'verification.json').write_text(json.dumps(dict(status='PASS',completed=count,signature=runtime.SIGNATURE,statistics_match=True,manual_field_validation='PENDING'),indent=2))
    print('PASS: statistics reproduced; no GPU or training was used.',flush=True)

if __name__=='__main__':main()

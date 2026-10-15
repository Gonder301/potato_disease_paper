from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
ROOT=Path('work/final_results/RESULTS_v4'); OUT=Path('outputs/Cierre_observaciones');OUT.mkdir(parents=True,exist_ok=True)
mapping=json.loads(Path('work/closure/leaf-map.json').read_text());mf=pd.read_csv(ROOT/'manifest.csv');df=mf[mf.domain=='potato_lab'].copy()
def resolve(rel):
 name=Path(rel).stem.split('___',1)[-1].lower().strip();cat=Path(rel).parent.name
 candidates=sorted(set(x for x in mapping.get(name,[]) if x.split(':::')[0]==cat and x.split(':::')[-1] not in {'','nan'}))
 return name,candidates[0] if len(candidates)==1 else '',len(candidates)
df[['reference_key','leaf_id','reference_candidates']]=pd.DataFrame([resolve(p) for p in df.relative_path],index=df.index)
df.to_csv(OUT/'plantvillage_referencia_hojas.csv',index=False)
matched=df[df.leaf_id!=''].merge(pd.read_csv(ROOT/'data/potato_lab/clusters.csv'),on='image_id',validate='one_to_one')
assert matched.image_id.is_unique
def comb(x):return (x*(x-1)//2).sum()
rows=[]
for tau in [.1,.15,.2]:
 c=f'tau_{tau}';tab=pd.crosstab(matched[c],matched.leaf_id);tp=int(comb(tab.to_numpy()));pred=int(comb(tab.sum(axis=1)));actual=int(comb(tab.sum(axis=0)))
 rows.append(dict(tau=tau,covered_images=len(matched),leaf_ids=matched.leaf_id.nunique(),predicted_positive_pairs=pred,reference_positive_pairs=actual,tp=tp,fp=pred-tp,fn=actual-tp,pair_precision=tp/pred if pred else None,pair_recall=tp/actual if actual else None,identity_purity=tab.max(axis=1).sum()/len(matched)))
stats=pd.DataFrame(rows);stats.to_csv(OUT/'validacion_grupos_identidad.csv',index=False)
ex=[]
for p in (ROOT/'data/potato_lab').glob('split_*.csv'):
 s=pd.read_csv(p).merge(df[['image_id','leaf_id']],on='image_id',validate='one_to_one');strategy,seed=p.stem.split('_')[1:]
 train=s[(s.split=='train')&(s.leaf_id!='')];test=s[s.split=='test'];known=test[test.leaf_id!='']; exposed=known.leaf_id.isin(train.leaf_id)
 ex.append(dict(strategy=strategy,seed=int(seed),test_total=len(test),test_with_reference=len(known),exposed=int(exposed.sum()),reference_exposure_percent=100*exposed.mean(),lower_percent_all_test=100*exposed.sum()/len(test),upper_percent_all_test=100*(exposed.sum()+len(test)-len(known))/len(test)))
pd.DataFrame(ex).to_csv(OUT/'exposicion_identidad_laboratorio.csv',index=False)
report=dict(images_total=len(df),images_reference=len(matched),coverage_percent=100*len(matched)/len(df),missing=df[df.leaf_id==''].image_id.tolist(),unique_leaf_ids=matched.leaf_id.nunique(),source='https://github.com/spMohanty/PlantVillage-Dataset',commit=json.loads(Path('work/closure/pv_tree.json').read_text())['sha'],map_sha256=hashlib.sha256(Path('work/closure/leaf-map.json').read_bytes()).hexdigest(),reference_type='Published leaf identity metadata; not newly performed manual near-copy annotation')
(OUT/'referencia_procedencia.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='missing'},indent=2));print(stats.to_string(index=False));print(pd.DataFrame(ex).groupby('strategy').reference_exposure_percent.agg(['mean','min','max']).to_string());print('missing',df[df.leaf_id==''].relative_path.head().tolist())

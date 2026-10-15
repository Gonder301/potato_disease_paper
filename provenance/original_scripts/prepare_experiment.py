import sys,json,hashlib,shutil,time,zipfile
from pathlib import Path
sys.path.insert(0,str(Path('work/qa_packages').resolve()));sys.path.insert(0,str(Path('work').resolve()))
import numpy as np,pandas as pd
import rebuild_v42 as base
from sklearn.cluster import AgglomerativeClustering
ROOT=Path('work/final_results/RESULTS_v4'); OUT=Path('outputs/Cierre_observaciones'); BUNDLE=OUT/'estado_inicial';BUNDLE.mkdir(parents=True,exist_ok=True)
cfg=base.DEFAULT.copy();cfg['split_solver_seconds']=90
old=pd.read_csv(ROOT/'manifest.csv'); reference=pd.read_csv(OUT/'plantvillage_referencia_hojas.csv')
lab=old[old.domain=='potato_lab'].copy().reset_index(drop=True);lab['domain']='potato_lab_leaf_id'
lab=lab.merge(reference[['image_id','leaf_id']],on='image_id',validate='one_to_one')
assert lab.leaf_id.notna().all() and lab.leaf_id.nunique()==538
lab['evaluation_group']=pd.factorize(lab.leaf_id,sort=True)[0];lab['group_basis']='published_leaf_identity'
field=old[old.source=='warcoder/potato-leaf-disease-dataset'].copy().reset_index(drop=True);field['domain']='potato_field_single_source'
field['label']=field.label_id.map({0:'healthy',1:'fungi',2:'phytophthora'});field['leaf_id']=''
cache=np.load(ROOT/'data/potato_field/embeddings.npz',allow_pickle=False);lookup={str(x):i for i,x in enumerate(cache['image_id'])};z=cache['z'][[lookup[i] for i in field.image_id]]
field['evaluation_group']=AgglomerativeClustering(n_clusters=None,metric='cosine',linkage='average',distance_threshold=.1).fit_predict(z);field['group_basis']='DINOv2_average_cosine_0.10_unvalidated_identity'
assert len(field)==1296 and field.source.nunique()==1
manifest=pd.concat([lab,field],ignore_index=True);manifest.to_csv(BUNDLE/'manifest.csv',index=False)
rows=[];reuse=[]
for domain,df in manifest.groupby('domain',sort=False):
 df=df.reset_index(drop=True);dest=BUNDLE/'data'/domain;dest.mkdir(parents=True,exist_ok=True)
 for seed in cfg['seeds']:
  for strategy in ['random','group']:
   path=dest/f'split_{strategy}_{seed}.csv'
   if path.exists():
    saved=pd.read_csv(path);assert saved.image_id.tolist()==df.image_id.tolist();parts=tuple(np.flatnonzero(saved.split.eq(k)) for k in ['train','val','test'])
   else:
    started=time.monotonic();print(f'PREPARANDO {domain} {strategy} semilla={seed}',flush=True)
    parts=base.random_split(df.label_id.to_numpy(),seed) if strategy=='random' else base.grouped_split(df.label_id.to_numpy(),df.evaluation_group.to_numpy(),seed,cfg)
    saved=df[['image_id','source','label','label_id','sha256','evaluation_group','group_basis','leaf_id']].copy();saved['split']=''
    for k,part in zip(['train','val','test'],parts):saved.loc[part,'split']=k
    saved.to_csv(path,index=False);print(f'OK {time.monotonic()-started:.1f}s {[len(p) for p in parts]}',flush=True)
   base.verify_split(parts,df.label_id.to_numpy(),df.evaluation_group.to_numpy() if strategy=='group' else None)
   for index,(name,part) in enumerate(zip(['train','val','test'],parts)):
    co=np.bincount(df.label_id.to_numpy()[part],minlength=3);total=np.bincount(df.label_id,minlength=3);target=[.7,.15,.15][index]
    assert abs(len(part)/len(df)-target)<=.02+1e-9 and max(abs(co/total-target))<=.03+1e-9
    rows.append(dict(domain=domain,seed=seed,strategy=strategy,part=name,n=len(part),class_0=int(co[0]),class_1=int(co[1]),class_2=int(co[2])))
   if domain=='potato_lab_leaf_id' and strategy=='random':
    orig=pd.read_csv(ROOT/'data/potato_lab'/path.name);assert orig.image_id.tolist()==saved.image_id.tolist();assert orig.split.tolist()==saved.split.tolist();assert orig.label_id.tolist()==saved.label_id.tolist()
pd.DataFrame(rows).to_csv(OUT/'particiones_nuevo_experimento.csv',index=False)
scientific={k:cfg[k] for k in ['models','seeds','epochs','batch_size','image_size','lr','weight_decay','mixed_precision','pretrained','split_tolerance','class_split_tolerance']}
scientific.update(version='5.0.0',domain_labels={'potato_lab_leaf_id':['healthy','early_blight','late_blight'],'potato_field_single_source':['healthy','fungi','phytophthora']},grouping={'potato_lab_leaf_id':'published_leaf_identity','potato_field_single_source':'DINOv2_average_cosine_0.10_unvalidated_identity'},reference_commit=json.loads((OUT/'referencia_procedencia.json').read_text())['commit'],primary_tests=24,study_type='revision_after_v42_findings_not_preregistered',expected=240,previous_random_runs_reused=60,new_runs_required=180)
files={str(p.relative_to(BUNDLE).as_posix()):base.file_hash(p) for p in BUNDLE.rglob('*.csv')}
signature=base.digest(dict(scientific=scientific,input_files=files));base.atomic_json(BUNDLE/'project_state.json',dict(version='5.0.0',signature=signature,scientific=scientific,input_files=files))
for model in cfg['models']:
 for seed in cfg['seeds']:
  source=ROOT/'runs'/f'potato_lab__{model}__random__{seed}';target=BUNDLE/'runs'/f'potato_lab_leaf_id__{model}__random__{seed}'
  target.mkdir(parents=True,exist_ok=True)
  for name in ['history.csv','predictions.csv']:shutil.copy2(source/name,target/name)
  info=json.loads((source/'complete.json').read_text());assert base.file_hash(target/'predictions.csv')==info['predictions_sha256']
  base.atomic_json(target/'source_complete_v42.json',info);info['domain']='potato_lab_leaf_id';info['signature']=signature
  info['reuse_provenance']=dict(source_signature=json.loads((ROOT/'project_state.json').read_text())['signature'],reason='Identical laboratory random train validation test IDs labels and training implementation; predictions unchanged')
  base.atomic_json(target/'complete.json',info);reuse.append(dict(model=model,seed=seed,sha256=info['predictions_sha256']))
pd.DataFrame(reuse).to_csv(BUNDLE/'reuse_audit.csv',index=False)
shutil.copy2(ROOT/'environment.json',BUNDLE/'origin_environment.json')
shutil.copy2(OUT/'referencia_procedencia.json',BUNDLE/'reference_provenance.json')
zipout=OUT/'Datos_y_60_corridas_reutilizables_v5.zip'
with zipfile.ZipFile(zipout,'w',zipfile.ZIP_DEFLATED) as z:
 for p in BUNDLE.rglob('*'):
  if p.is_file():z.write(p,'RESULTS_closure/'+p.relative_to(BUNDLE).as_posix())
print('READY',signature,'60 splits, 60 reused runs, 180 new required',zipout,flush=True)

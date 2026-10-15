"""Frozen v5 evaluation. Uses the unchanged v4.2 training implementation."""
import json, time, shutil, zipfile, traceback, sys, os, gc, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import engine_v5 as base

SIGNATURE = '773af3d89fa82d512df8da024e151be6df88ed9a06bb4ff40797612038c6824c'

def validate(root, signature=SIGNATURE):
    state=json.loads((root/'project_state.json').read_text())
    if state['version']!='5.0.0' or state['signature']!=signature:
        raise ValueError('Estado incompatible: se requiere este experimento v5, no RESULTS_v4.')
    if base.digest(dict(scientific=state['scientific'],input_files=state['input_files']))!=signature:
        raise ValueError('La especificación del experimento cambió.')
    for rel, expected in state['input_files'].items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root.resolve()) or base.file_hash(p)!=expected:
            raise ValueError(f'Archivo alterado o ausente: {rel}')
    df=pd.read_csv(root/'manifest.csv'); plans={}; sci=state['scientific']
    if not df.image_id.is_unique: raise ValueError('IDs de imagen repetidos.')
    for domain,labels in sci['domain_labels'].items():
        d=df[df.domain.eq(domain)].reset_index(drop=True)
        if d.empty or not d.label.equals(d.label_id.map(dict(enumerate(labels)))):
            raise ValueError('Etiquetas incompatibles con el protocolo.')
        if domain=='potato_field_single_source' and (len(d)!=1296 or d.source.nunique()!=1):
            raise ValueError('El control de fuente de campo no coincide.')
        if domain=='potato_lab_leaf_id' and (len(d)!=2152 or d.leaf_id.isna().any() or d.leaf_id.nunique()!=538):
            raise ValueError('Referencia de identidad incompleta.')
        local={}
        for seed in sci['seeds']:
            for strategy in ['random','group']:
                s=pd.read_csv(root/'data'/domain/f'split_{strategy}_{seed}.csv')
                if s.image_id.tolist()!=d.image_id.tolist() or s.label_id.tolist()!=d.label_id.tolist():
                    raise ValueError('Orden de IDs o etiquetas de partición alterado.')
                parts=tuple(np.flatnonzero(s.split.eq(k)) for k in ['train','val','test'])
                base.verify_split(parts,d.label_id.to_numpy(),d.evaluation_group.to_numpy() if strategy=='group' else None)
                for part,target in zip(parts,[.7,.15,.15]):
                    counts=np.bincount(d.label_id.to_numpy()[part],minlength=3)
                    if abs(len(part)/len(d)-target)>.020000001 or max(abs(counts/np.bincount(d.label_id,minlength=3)-target))>.030000001:
                        raise ValueError('Partición fuera de tolerancia.')
                local[(strategy,seed)]=parts
        plans[domain]=(d,local)
    allowed=set()
    completed=0
    for domain,(d,local) in plans.items():
        for name in sci['models']:
            for (strategy,seed),parts in local.items():
                rid=f'{domain}__{name}__{strategy}__{seed}';allowed.add(rid)
                rd=root/'runs'/rid
                if base.finished(rd,signature,d.image_id.iloc[parts[2]]):
                    info=json.loads((rd/'complete.json').read_text()); pred=pd.read_csv(rd/'predictions.csv')
                    if any(info[k]!=v for k,v in dict(domain=domain,model=name,strategy=strategy,seed=seed,epochs=sci['epochs']).items()):
                        raise ValueError('Marcador de corrida incompatible.')
                    prob=pred[[f'prob_{c}' for c in sci['domain_labels'][domain]]].to_numpy()
                    if not np.isfinite(prob).all() or (prob<0).any() or not np.allclose(prob.sum(1),1,atol=1e-5) or not np.array_equal(prob.argmax(1),pred.y_pred):
                        raise ValueError('Probabilidades inválidas.')
                    if pred.y_true.tolist()!=d.label_id.iloc[parts[2]].tolist(): raise ValueError('Verdad de test alterada.')
                    for k,v in base.metric_values(pred.y_true,pred.y_pred).items():
                        if not np.isclose(v,info['metrics'][k],atol=1e-12): raise ValueError('Métrica no reproducible.')
                    completed+=1
    if any(p.parent.name not in allowed for p in (root/'runs').glob('*/complete.json')):
        raise ValueError('Resultados de otro experimento en la carpeta.')
    return state,plans,completed

def resolve_images(plans,input_root,log):
    index={}
    log.log('VERIFICANDO imágenes: indexación de archivos; no extrae embeddings ni recalcula particiones.')
    for p in input_root.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}:
            index.setdefault(p.name,[]).append(p)
    total=sum(len(d) for d,_ in plans.values()); done=0
    for domain,(d,_) in plans.items():
        resolved=[]
        for row in d.itertuples():
            old=Path(row.path); rel=Path(row.relative_path)
            candidates=[old] if old.is_file() and old.resolve().is_relative_to(input_root.resolve()) else []
            candidates+=index.get(rel.name,[])
            chosen=None
            for p in dict.fromkeys(candidates):
                if base.file_hash(p)==row.sha256:
                    with Image.open(p) as im: im.verify()
                    chosen=p;break
            if chosen is None: raise FileNotFoundError(f'Imagen ausente o bytes distintos: {row.source}/{row.relative_path}')
            resolved.append(str(chosen)); done+=1
            if done%250==0 or done==total: log.log(f'IMAGENES VERIFICADAS {done}/{total}: SHA-256 y decodificación OK.')
        d['path']=resolved

def export(root,log):
    target=root.parent/'results_closure_v5.zip'; temp=target.with_suffix('.tmp.zip')
    log.log('EXPORTANDO resultados y checkpoint pendiente; conservar ZIP para continuar.')
    with zipfile.ZipFile(temp,'w',allowZip64=True) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and '.tmp' not in p.name:
                z.write(p,'RESULTS_closure/'+p.relative_to(root).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix=='.pt' else zipfile.ZIP_DEFLATED)
    os.replace(temp,target)
    log.log(f'ZIP LISTO: {target}, {target.stat().st_size/1e6:.1f} MB.')

def initialize(root, input_root, resume_dir=''):
    if (root/'project_state.json').exists(): return
    if root.exists() and any(root.iterdir()): raise ValueError('Carpeta de salida no vacía sin estado válido.')
    if resume_dir:
        source=Path(resume_dir)
    else:
        candidates=[]
        for p in input_root.rglob('project_state.json'):
            try:
                if json.loads(p.read_text()).get('signature')==SIGNATURE: candidates.append(p.parent)
            except (ValueError,OSError): pass
        if len(candidates)!=1: raise ValueError(f'Se necesita exactamente un estado v5 conectado (encontrados {len(candidates)}). Define RESUME_DIR explícitamente si hay varios.')
        source=candidates[0]
    validate(source)
    shutil.copytree(source,root,dirs_exist_ok=True)

def run(output_root='/kaggle/working/RESULTS_closure',input_root='/kaggle/input',resume_dir='',session_hours=8.,reserve_minutes=15):
    root=Path(output_root); inputs=Path(input_root)
    initialize(root,inputs,resume_dir)
    log=base.Monitor(root,60);log.start();error=None;paused=False;count=0;expected=240;state=None
    started=time.monotonic(); deadline=started+max(0,session_hours*3600-reserve_minutes*60)
    try:
        log.log('PREVALIDACION v5: referencias, particiones congeladas y resultados reutilizados.')
        state,plans,count=validate(root); sci=state['scientific'];expected=sci['expected']
        cfg={**base.DEFAULT,**{k:v for k,v in sci.items() if k in base.DEFAULT},'workers':0}
        log.log(f'PLAN {count}/{expected} completas verificadas; faltan {expected-count}. Presupuesto {session_hours} h; reserva {reserve_minutes} min.')
        base.summarize(root,cfg,SIGNATURE,log)
        if count==expected: return
        env=dict(python=sys.version,packages={k:importlib.metadata.version(k) for k in ['torch','torchvision','timm','numpy','pandas','scikit-learn','scipy','ImageHash']})
        base.atomic_json(root/'environment.json',env)
        origin=json.loads((root/'origin_environment.json').read_text())
        for k in origin['packages']:
            if env['packages'][k]!=origin['packages'][k]: raise RuntimeError(f'Entorno incompatible para reutilizar corridas: {k}={env["packages"][k]}, esperado {origin["packages"][k]}. Corregir antes de entrenar.')
        if not torch.cuda.is_available(): raise RuntimeError('GPU no disponible: activar GPU en Kaggle antes de ejecutar.')
        device=torch.device('cuda');log.log(f'GPU: {torch.cuda.get_device_name(0)}; usa una GPU; {torch.cuda.device_count()} visibles.')
        resolve_images(plans,inputs,log)
        for name in sci['models']:
            if time.monotonic()>=deadline: raise base.PauseRun('Presupuesto agotado durante prevalidación.')
            log.log(f'PREVALIDACION arquitectura {name}: forward y backward.')
            base.seed_all(42);m=base.build_model(name,cfg,device,pretrained=False)
            with torch.autocast(device_type='cuda',enabled=cfg['mixed_precision']):
                out=m(torch.zeros(2,3,224,224,device=device));loss=torch.nn.functional.cross_entropy(out,torch.tensor([0,1],device=device))
            if out.shape!=(2,3) or not torch.isfinite(loss): raise ValueError('Salida de modelo inválida.')
            loss.backward();del m,out,loss;gc.collect();torch.cuda.empty_cache();log.log(f'ARQUITECTURA OK: {name}')
        new_completed=0;training_start=time.monotonic()
        for name in sci['models']:
            for domain,(d,plans_domain) in plans.items():
                base.CLASSES=sci['domain_labels'][domain]
                for seed in sci['seeds']:
                    for strategy in ['random','group']:
                        parts=plans_domain[(strategy,seed)];rd=root/'runs'/f'{domain}__{name}__{strategy}__{seed}'
                        if base.finished(rd,SIGNATURE,d.image_id.iloc[parts[2]]): continue
                        base.train_one(d,parts,domain,name,strategy,seed,rd,cfg,SIGNATURE,device,log,deadline)
                        count+=1;new_completed+=1
                        eta=(time.monotonic()-training_start)/new_completed*(expected-count)/3600
                        base.atomic_json(root/'session_result.json',dict(status='RUNNING',completed=count,expected=expected,scientific_results=True,error=None))
                        log.log(f'PROGRESO GLOBAL {count}/{expected} ({100*count/expected:.1f}%); nuevas esta sesión={new_completed}; ETA aproximada={eta:.1f} h, varía según arquitectura.')
    except base.PauseRun as e:
        paused=True;log.log(f'PAUSA CONTROLADA: {e}')
    except (Exception,KeyboardInterrupt) as e:
        error=e;log.log(f'ERROR {type(e).__name__}: {e}');(root/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
    finally:
        try:
            if state:
                try: count=base.summarize(root,cfg,SIGNATURE,log)
                except Exception as e:
                    if error is None: error=e
                    log.log(f'ERROR al resumir: {e}')
            status='ERROR' if error else ('COMPLETE' if count==expected else 'PAUSED')
            base.atomic_json(root/'session_result.json',dict(status=status,completed=count,expected=expected,scientific_results=error is None,error=str(error) if error else None))
            export(root,log);log.log(f'FIN {status}: {count}/{expected}.')
        finally: log.close()
    if error: raise error

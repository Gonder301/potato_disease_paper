"""Motor reproducible para Kaggle. El notebook incorpora este archivo completo."""
from __future__ import annotations
import os, sys, json, time, math, random, hashlib, shutil, threading, traceback, zipfile, gc
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm, imagehash
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from scipy.stats import wilcoxon

VERSION = "4.2.0"
CLASSES = ["healthy", "early_blight", "late_blight"]
MEAN, STD = (.485, .456, .406), (.229, .224, .225)
DEFAULT = dict(
    input_root="/kaggle/input", output_root="/kaggle/working/RESULTS_v4",
    resume_dir="", session_hours=8.0, reserve_minutes=15,
    models=["resnet50", "mobilenetv3_large_100", "tf_efficientnetv2_s", "vit_base_patch16_224"],
    seeds=[42,0,1,7,123,77,88,99,101,2024,11,19,23,31,47],
    epochs=10, batch_size=32, embedding_batch_size=16, image_size=224,
    lr=3e-4, weight_decay=1e-4, tau=.10, taus=[.10,.15,.20],
    mixed_precision=True, require_gpu=True, workers=0, pretrained=True,
    heartbeat_seconds=60, batch_log_seconds=60, bootstrap_samples=5000,
    group_candidates=256, split_tolerance=.02, class_split_tolerance=.03, split_solver_seconds=30, preflight=True,
    # Especificación de fuentes, nombres de carpeta exactos (sin adivinar por ruta completa).
    # Las dos equivalencias Fungi/Phytopthora son las del protocolo previo: verificar su semántica.
    sources=[
        dict(slug="arjuntejaswi/plant-village", domain="potato_lab",
             labels={"potato___healthy":"healthy","potato___early_blight":"early_blight",
                     "potato___late_blight":"late_blight"}),
        dict(slug="warcoder/potato-leaf-disease-dataset", domain="potato_field",
             labels={"healthy":"healthy","fungi":"early_blight","phytopthora":"late_blight",
                     "early_blight":"early_blight","late_blight":"late_blight"}),
        dict(slug="nirmalsankalana/potato-leaf-healthy-and-late-blight", domain="potato_field",
             labels={"healthy":"healthy"})],
    # QA local exclusivamente: nunca usar para resultados del artículo.
    smoke_test=False, fixture_manifest="", domain_filter=[],
)
def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",",":"), default=str)
def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()
def file_hash(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def atomic_json(p, obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+".tmp"); t.write_text(canonical(obj),encoding="utf-8"); os.replace(t,p)
def atomic_csv(p, df):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+".tmp"); df.to_csv(t,index=False); os.replace(t,p)
def atomic_torch(p, obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(".tmp"); torch.save(obj,t); os.replace(t,p)
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

class PauseRun(Exception): pass
class Monitor:
    def __init__(self, root, seconds=60):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.started=time.monotonic(); self.last=self.started; self.phase="inicio"
        self.lock=threading.RLock(); self.event=threading.Event(); self.seconds=seconds
    def log(self,msg,advance=True):
        with self.lock:
            now=time.monotonic()
            if advance: self.phase=msg; self.last=now
            line=f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] +{(now-self.started)/60:.1f} min | {msg}"
            print(line,flush=True)
            with open(self.root/"execution.log","a",encoding="utf-8") as f: f.write(line+"\n")
            atomic_json(self.root/"status.json",dict(elapsed_seconds=now-self.started,
                last_progress_seconds_ago=now-self.last, phase=self.phase, message=msg))
    def beat(self):
        while not self.event.wait(self.seconds):
            age=time.monotonic()-self.last
            self.log(f"LATIDO: proceso responde; última operación: {self.phase}; sin hito nuevo hace {age:.0f} s. No certifica avance.",False)
    def start(self):
        self.thread=threading.Thread(target=self.beat,daemon=True); self.thread.start()
    def close(self):
        self.event.set(); self.thread.join(timeout=2)

def normalize(s):
    return s.lower().replace(" ","_").replace("-","_")
def scan_manifest(cfg,log):
    if cfg["fixture_manifest"]:
        if not cfg["smoke_test"]: raise ValueError("fixture_manifest requiere smoke_test.")
        df=pd.read_csv(cfg["fixture_manifest"])
        return df.sort_values("image_id").reset_index(drop=True)
    root=Path(cfg["input_root"])
    if not root.exists(): raise FileNotFoundError(f"No existe {root}; adjunta datasets.")
    rows=[]; skipped=[]; found_sources=0
    for spec in cfg["sources"]:
        slug=spec["slug"]; base=slug.split("/")[-1]
        roots=sorted(p for p in root.rglob(base) if p.is_dir())
        if len(roots)>1:
            raise ValueError(f"Varias versiones de {slug}: {roots}. Adjunta solo una.")
        if not roots:
            log.log(f"Fuente no adjunta: {slug}"); continue
        found_sources+=1; folder=roots[0]
        images=sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"} and p.is_file())
        log.log(f"MANIFIESTO {slug}: {len(images)} archivos; leyendo etiquetas y SHA-256.")
        for i,p in enumerate(images):
            rel=p.relative_to(folder).as_posix()
            labels={spec["labels"][normalize(x)] for x in p.relative_to(folder).parts[:-1] if normalize(x) in spec["labels"]}
            if not labels: skipped.append(dict(source=slug,relative_path=rel)); continue
            if len(labels)!=1: raise ValueError(f"Etiquetas ambiguas: {p}")
            label=labels.pop()
            with Image.open(p) as im: im.verify()
            h=file_hash(p)
            rows.append(dict(image_id=digest([slug,rel])[:24],path=str(p),relative_path=rel,
                source=slug,domain=spec["domain"],label=label,label_id=CLASSES.index(label),sha256=h))
            if (i+1)%500==0: log.log(f"MANIFIESTO {slug}: {i+1}/{len(images)} leídos.")
    if not rows: raise ValueError("Sin imágenes de las fuentes configuradas. Revisa sources; no se aceptan etiquetas inferidas del nombre del dataset.")
    df=pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    if cfg["domain_filter"]: df=df[df.domain.isin(cfg["domain_filter"])].reset_index(drop=True)
    if df.empty: raise ValueError("domain_filter dejó el manifiesto vacío.")
    if df.image_id.duplicated().any(): raise ValueError("IDs repetidos.")
    conflict=df.groupby("sha256").label.nunique()
    if (conflict>1).any(): raise ValueError("Mismo archivo con etiquetas distintas; corregir datos antes de entrenar.")
    atomic_csv(Path(cfg["output_root"])/"skipped_images.csv",pd.DataFrame(skipped,columns=["source","relative_path"]))
    for name,g in df.groupby("domain"):
        cnt=g.label.value_counts()
        if set(cnt.index)!=set(CLASSES) or cnt.min()<7:
            raise ValueError(f"{name}: se necesitan las tres clases y >=7 imágenes/clase; encontradas {cnt.to_dict()}.")
    log.log(f"MANIFIESTO: {len(df)} imágenes; {found_sources} fuentes; {df.sha256.duplicated(False).sum()} archivos en grupos exactos repetidos.")
    print(df.groupby(["domain","source","label"]).size().to_string(),flush=True)
    return df

class Images(Dataset):
    def __init__(self,df,tf): self.paths=df.path.tolist(); self.y=df.label_id.to_numpy(); self.tf=tf
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as im: x=self.tf(im.convert("RGB"))
        return x,int(self.y[i])
def loader(df,tf,cfg,shuffle=False,seed=0):
    return DataLoader(Images(df,tf),batch_size=cfg["batch_size"],shuffle=shuffle,
        num_workers=cfg["workers"],pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),drop_last=False)
def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def build_model(name,cfg,device,pretrained=None):
    if cfg["smoke_test"]:
        return nn.Sequential(nn.Conv2d(3,6,3,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(6,3)).to(device)
    return timm.create_model(name,pretrained=cfg["pretrained"] if pretrained is None else pretrained,
        num_classes=3).to(device)
def transforms(cfg,train=False):
    ops=[T.Resize((cfg["image_size"],cfg["image_size"]))]
    if train: ops += [T.RandomHorizontalFlip(),T.RandomRotation(10)]
    return T.Compose(ops+[T.ToTensor(),T.Normalize(MEAN,STD)])

def prepare_embeddings(df,directory,cfg,device,log):
    directory.mkdir(parents=True,exist_ok=True); cache=directory/"embeddings.npz"
    if cache.exists():
        with np.load(cache,allow_pickle=False) as a:
            if list(a["image_id"])!=df.image_id.tolist(): raise ValueError("Cache de embeddings incompatible.")
            z=a["z"].copy(); ph=a["phash"].copy()
        log.log(f"EMBEDDINGS {df.domain.iloc[0]}: caché válido, omitido."); return z,ph
    log.log(f"EMBEDDINGS {df.domain.iloc[0]}: {'QA sintético' if cfg['smoke_test'] else 'cargando DINOv2 (puede descargar pesos)'}")
    if cfg["smoke_test"]:
        rng=np.random.default_rng(31); z=rng.normal(size=(len(df),12))
        z/=np.linalg.norm(z,axis=1,keepdims=True)
    else:
        dino=timm.create_model("vit_small_patch14_dinov2.lvd142m",pretrained=True,num_classes=0).to(device).eval()
        tf=T.Compose([T.Resize((518,518)),T.ToTensor(),T.Normalize(MEAN,STD)])
        dl=DataLoader(Images(df,tf),batch_size=cfg["embedding_batch_size"],num_workers=cfg["workers"])
        chunks=[]; last=time.monotonic()
        with torch.no_grad():
            for b,(x,_) in enumerate(dl,1):
                chunks.append(dino(x.to(device)).float().cpu().numpy())
                if b==1 or b==len(dl) or time.monotonic()-last>=cfg["batch_log_seconds"]:
                    log.log(f"EMBEDDINGS lote {b}/{len(dl)}"); last=time.monotonic()
        z=np.concatenate(chunks); z/=np.linalg.norm(z,axis=1,keepdims=True).clip(min=1e-12)
        del dino,chunks; gc.collect()
        if device.type=="cuda": torch.cuda.empty_cache()
    ph=[]
    for i,p in enumerate(df.path):
        with Image.open(p) as im: ph.append(imagehash.phash(im.convert("RGB")).hash.flatten())
        if (i+1)%1000==0: log.log(f"pHash {i+1}/{len(df)}")
    ph=np.asarray(ph,dtype=np.uint8)
    tmp=cache.with_suffix(".tmp.npz")
    np.savez_compressed(tmp,image_id=df.image_id.to_numpy(dtype=str),z=z,phash=ph)
    os.replace(tmp,cache); log.log(f"EMBEDDINGS y pHash guardados: {len(z)} imágenes.")
    return z,ph

"""Partición 70/15/15 por imágenes con grupos indivisibles y diagnóstico explícito."""
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix

class SplitInfeasible(ValueError):
    """El solver demostró que las restricciones no admiten una partición."""
class SplitSearchLimit(RuntimeError):
    """La búsqueda terminó sin encontrar solución; no demuestra inviabilidad."""

def grouped_split(y,groups,seed,cfg):
    y=np.asarray(y,dtype=int); groups=np.asarray(groups)
    if len(y)!=len(groups) or set(y)!={0,1,2}:
        raise ValueError("Se necesitan etiquetas 0,1,2 y un grupo por imagen.")
    unique,inverse=np.unique(groups,return_inverse=True); n_groups=len(unique)
    if n_groups<3: raise SplitInfeasible("Menos de tres grupos indivisibles.")
    counts=np.zeros((n_groups,3),dtype=int)
    np.add.at(counts,(inverse,y),1)
    sizes=counts.sum(1); n=len(y); target=np.array([.70,.15,.15])
    tolerance=float(cfg["split_tolerance"])
    lo=np.ceil((target-tolerance).clip(0)*n-1e-9)
    hi=np.floor((target+tolerance).clip(max=1)*n+1e-9)
    if sizes.max()>hi.max():
        raise SplitInfeasible(f"Un grupo tiene {sizes.max()}/{n} imágenes; supera el máximo admisible {int(hi.max())}.")
    for c in range(3):
        if np.count_nonzero(counts[:,c])<3:
            raise SplitInfeasible(f"La clase {c} aparece en menos de tres grupos indivisibles.")

    class_tolerance=float(cfg.get("class_split_tolerance",.03))
    class_totals=counts.sum(0)
    class_lo=np.maximum(1,np.ceil((target[:,None]-class_tolerance).clip(0)*class_totals-1e-9))
    class_hi=np.floor((target[:,None]+class_tolerance).clip(max=1)*class_totals+1e-9)
    for c in range(3):
        if counts[:,c].max()>class_hi[:,c].max():
            raise SplitInfeasible(f"Clase {c}: un grupo tiene {counts[:,c].max()}/{class_totals[c]} imágenes, superior al máximo por clase admisible {int(class_hi[:,c].max())}.")
    # x[g,s] binary; d[s,k] nonnegative absolute deviation for total and each class.
    features=np.column_stack([sizes,counts]); totals=features.sum(0)
    nx=3*n_groups; nv=nx+12
    matrix=lil_matrix((n_groups+12+24,nv),dtype=float)
    lower=np.full(matrix.shape[0],-np.inf); upper=np.full(matrix.shape[0],np.inf)
    row=0
    for g in range(n_groups):
        matrix[row,3*g:3*g+3]=1; lower[row]=upper[row]=1; row+=1
    for s in range(3):
        indices=np.arange(n_groups)*3+s
        for k in range(4):
            matrix[row,indices]=features[:,k]
            lower[row]=lo[s] if k==0 else class_lo[s,k-1]
            upper[row]=hi[s] if k==0 else class_hi[s,k-1]
            row+=1
    for s in range(3):
        indices=np.arange(n_groups)*3+s
        for k in range(4):
            wanted=target[s]*totals[k]; dev=nx+4*s+k
            matrix[row,indices]=features[:,k]; matrix[row,dev]=-1
            upper[row]=wanted; row+=1
            matrix[row,indices]=-features[:,k]; matrix[row,dev]=-1
            upper[row]=-wanted; row+=1
    objective=np.zeros(nv)
    objective[nx:]=np.tile(1/np.maximum(totals,1),3)
    # Seed affects tie-breaking, never feasibility, labels or target proportions.
    objective[:nx]=np.random.default_rng(seed).uniform(0,1e-7/n_groups,nx)
    integrality=np.zeros(nv); integrality[:nx]=1
    bound_hi=np.full(nv,np.inf); bound_hi[:nx]=1
    solved=milp(c=objective,integrality=integrality,
        bounds=Bounds(np.zeros(nv),bound_hi),
        constraints=LinearConstraint(matrix.tocsc(),lower,upper),
        options=dict(time_limit=float(cfg.get("split_solver_seconds",30)),mip_rel_gap=.01))
    if solved.status==2:
        raise SplitInfeasible(
            f"No existe partición 70/15/15 con tolerancia ±{100*tolerance:.0f} puntos, "
            f"grupos intactos y balance por clase. n={n}, grupos={n_groups}, "
            f"mayor grupo={sizes.max()}. No cambiar tau según exactitud del modelo.")
    if solved.x is None:
        raise SplitSearchLimit(f"Búsqueda sin solución verificada (status={solved.status}): {solved.message}")
    x=solved.x[:nx].reshape(n_groups,3)
    if not np.allclose(x,np.rint(x),atol=1e-5):
        raise SplitSearchLimit("El solver no devolvió asignaciones enteras verificables.")
    assignment=x.argmax(1)
    parts=tuple(np.flatnonzero(assignment[inverse]==s) for s in range(3))
    for s,p in enumerate(parts):
        if not lo[s]<=len(p)<=hi[s] or set(y[p])!={0,1,2}:
            raise SplitSearchLimit("La solución no superó la validación independiente.")
        observed=np.bincount(y[p],minlength=3)
        if np.any(observed<class_lo[s]) or np.any(observed>class_hi[s]):
            raise SplitSearchLimit("La solución no respeta el balance por clase.")
    return parts


def random_split(y,seed):
    ix=np.arange(len(y)); tr,r=train_test_split(ix,test_size=.3,stratify=y,random_state=seed)
    va,te=train_test_split(r,test_size=.5,stratify=np.asarray(y)[r],random_state=seed)
    return tuple(np.sort(k) for k in (tr,va,te))
def verify_split(parts,y,groups=None):
    tr,va,te=parts
    assert len(set(tr)&set(va))==len(set(tr)&set(te))==len(set(va)&set(te))==0
    assert sorted(np.concatenate(parts).tolist())==list(range(len(y)))
    for k in parts:
        if set(np.asarray(y)[k])!={0,1,2}: raise ValueError("Partición sin las tres clases.")
    if groups is not None:
        gs=[set(np.asarray(groups)[k]) for k in parts]
        assert not (gs[0]&gs[1] or gs[0]&gs[2] or gs[1]&gs[2])

def prepare_groups_splits(df,z,ph,directory,cfg,log):
    saved=directory/"clusters.csv"
    if saved.exists():
        g=pd.read_csv(saved)
        if g.image_id.tolist()!=df.image_id.tolist(): raise ValueError("Clusters incompatibles.")
        clusters={t:g[f"tau_{t}"].to_numpy() for t in cfg["taus"]}
        log.log("CLUSTERS: caché válido.")
    else:
        clusters={}
        for tau in cfg["taus"]:
            log.log(f"CLUSTERING average/cosine: tau={tau}, n={len(z)}. Etapa en CPU.")
            started=time.monotonic()
            clusters[tau]=AgglomerativeClustering(n_clusters=None,metric="cosine",linkage="average",distance_threshold=tau).fit_predict(z)
            log.log(f"CLUSTERING tau={tau}: {len(np.unique(clusters[tau]))} grupos en {time.monotonic()-started:.1f} s.")
        atomic_csv(saved,pd.DataFrame(dict(image_id=df.image_id,**{f"tau_{t}":g for t,g in clusters.items()})))
    diag=[]
    for t,g in clusters.items():
        sizes=np.bincount(g)
        diag.append(dict(tau=t,clusters=len(sizes),images=len(g),max_size=int(sizes.max()),
            singleton_image_percent=100*float(np.mean(sizes[g]==1))))
    atomic_csv(directory/"cluster_diagnostics.csv",pd.DataFrame(diag))
    size_rows=[]
    for t,g in clusters.items():
        sizes=np.bincount(g)
        for group in np.unique(g):
            mask=g==group
            cc=np.bincount(df.label_id.to_numpy()[mask],minlength=3)
            size_rows.append(dict(tau=t,group=int(group),images=int(mask.sum()),
                healthy=int(cc[0]),early_blight=int(cc[1]),late_blight=int(cc[2])))
        log.log(f"DIAGNOSTICO tau={t}: mayor grupo={sizes.max()}/{len(g)} ({100*sizes.max()/len(g):.1f}%).")
    atomic_csv(directory/"cluster_sizes_by_class.csv",pd.DataFrame(size_rows))

    plans={}
    for seed in cfg["seeds"]:
        for strategy in ["random","group"]:
            p=directory/f"split_{strategy}_{seed}.csv"
            if p.exists():
                saved_split=pd.read_csv(p)
                if saved_split.image_id.tolist()!=df.image_id.tolist(): raise ValueError("Split incompatible.")
                parts=tuple(np.flatnonzero(saved_split["split"].eq(s)) for s in ["train","val","test"])
            else:
                log.log(f"BUSCANDO SPLIT {df.domain.iloc[0]} {strategy} seed={seed}; límite solver={cfg['split_solver_seconds']} s.")
                try:
                    parts=random_split(df.label_id.to_numpy(),seed) if strategy=="random" else grouped_split(df.label_id.to_numpy(),clusters[cfg["tau"]],seed,cfg)
                except (SplitInfeasible,SplitSearchLimit) as exc:
                    atomic_json(directory/"split_failure.json",dict(error=type(exc).__name__,
                        message=str(exc),tau=cfg["tau"],seed=seed,strategy=strategy))
                    raise
                labels=np.full(len(df),"",dtype=object)
                for k,n in zip(parts,["train","val","test"]): labels[k]=n
                saved_split=df[["image_id","source","label","label_id","sha256"]].copy()
                saved_split["split"]=labels; saved_split["visual_group"]=clusters[cfg["tau"]]
                atomic_csv(p,saved_split)
            verify_split(parts,df.label_id.to_numpy(),clusters[cfg["tau"]] if strategy=="group" else None)
            target=np.array([.70,.15,.15])
            counts=np.bincount(df.label_id.to_numpy(),minlength=3)
            for part_index,indices in enumerate(parts):
                if abs(len(indices)/len(df)-target[part_index])>cfg["split_tolerance"]+1e-9:
                    raise ValueError("Split cacheado fuera de tolerancia global.")
                observed=np.bincount(df.label_id.to_numpy()[indices],minlength=3)
                if np.max(np.abs(observed/counts-target[part_index]))>cfg["class_split_tolerance"]+1e-9:
                    raise ValueError("Split cacheado fuera de tolerancia por clase.")
            plans[(strategy,seed)]=parts
            log.log(f"SPLIT {df.domain.iloc[0]} {strategy} semilla={seed}: train/val/test={list(map(len,parts))}; validado.")
    # Exact exposure is measurable without pretending visual clusters are verified duplicates.
    exposure=[]
    for (strategy,seed),(tr,va,te) in plans.items():
        exposed=df.sha256.iloc[te].isin(set(df.sha256.iloc[tr]))
        exposure.append(dict(strategy=strategy,seed=seed,test_images=len(te),
            exact_exposed_images=int(exposed.sum()),exact_exposure_rate=float(exposed.mean()),
            near_duplicate_exposure="NOT_MEASURED"))
    atomic_csv(directory/"exact_duplicate_exposure.csv",pd.DataFrame(exposure))
    # Uniform pair sample: descriptive manual audit only, no unsupported population recall/purity.
    rng=np.random.default_rng(42); pairs=set(); target=min(500,len(df)*(len(df)-1)//2)
    while len(pairs)<target:
        a,b=sorted(rng.choice(len(df),2,replace=False).tolist()); pairs.add((a,b))
    annotations=[]
    for a,b in sorted(pairs):
        annotations.append(dict(image_a=df.image_id.iloc[a],image_b=df.image_id.iloc[b],
            path_a=df.path.iloc[a],path_b=df.path.iloc[b],
            cosine_distance=float(np.clip(1-np.dot(z[a],z[b]),0,2)),
            phash_hamming=int(np.count_nonzero(ph[a]!=ph[b])),
            same_visual_cluster=int(clusters[cfg["tau"]][a]==clusters[cfg["tau"]][b]),
            same_sha256=int(df.sha256.iloc[a]==df.sha256.iloc[b]),manual_duplicate=""))
    atomic_csv(directory/"manual_pair_sample.csv",pd.DataFrame(annotations))
    return plans

def metric_values(y,p):
    return dict(accuracy=float(accuracy_score(y,p)),balanced_accuracy=float(balanced_accuracy_score(y,p)),
        macro_f1=float(f1_score(y,p,labels=[0,1,2],average="macro",zero_division=0)))
def finished(run_dir,signature,expected_ids=None):
    marker=run_dir/"complete.json"; pred=run_dir/"predictions.csv"
    if not marker.exists(): return False
    info=json.loads(marker.read_text())
    if info["signature"]!=signature or not pred.exists() or file_hash(pred)!=info["predictions_sha256"]:
        raise ValueError(f"Resultados corruptos o incompatibles: {run_dir}")
    if expected_ids is not None:
        got=pd.read_csv(pred)
        if got.image_id.tolist()!=list(expected_ids): raise ValueError("IDs de predicción no coinciden con test.")
    return True

def train_one(df,parts,domain,name,strategy,seed,run_dir,cfg,signature,device,log,deadline):
    tr,va,te=parts; run_dir.mkdir(parents=True,exist_ok=True)
    if finished(run_dir,signature,df.image_id.iloc[te]):
        log.log(f"OMITIDO: {domain}/{name}/{strategy}/{seed} ya terminado y verificado."); return True
    if time.monotonic()>=deadline: raise PauseRun("Presupuesto de sesión alcanzado antes de nueva corrida.")
    log.log(f"INICIO MODELO {domain}/{name}/{strategy}, semilla={seed}.")
    ckpt=run_dir/"last.pt"; state=None
    if ckpt.exists():
        log.log("REANUDAR: leyendo checkpoint seguro (modelo, optimizador, época y mejor modelo).")
        state=torch.load(ckpt,map_location="cpu",weights_only=True)
        if state["signature"]!=signature: raise ValueError("Checkpoint incompatible.")
    seed_all(seed)
    m=build_model(name,cfg,device,pretrained=False if state else None)
    opt=torch.optim.AdamW(m.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    amp=cfg["mixed_precision"] and device.type=="cuda"
    scaler=torch.amp.GradScaler("cuda",enabled=amp)
    start=0; best=-1.; best_state=None; history=[]
    if state:
        m.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"]); start=state["epoch"]; best=state["best"]
        best_state=state["best_model"]; history=state["history"]
        del state
        log.log(f"REANUDAR desde época {start+1}/{cfg['epochs']}; {start} épocas guardadas.")
    train_dl=loader(df.iloc[tr],transforms(cfg,True),cfg,True,seed)
    val_dl=loader(df.iloc[va],transforms(cfg),cfg)
    try:
        for epoch in range(start,cfg["epochs"]):
            if time.monotonic()>=deadline: raise PauseRun("Pausa antes de nueva época; checkpoint preservado.")
            epoch_seed=(seed*100003+epoch)%(2**32-1)
            seed_all(epoch_seed); train_dl.generator.manual_seed(epoch_seed)
            begin=time.monotonic(); log.log(f"EPOCA {epoch+1}/{cfg['epochs']} | {name} {strategy} seed={seed}: {len(train_dl)} lotes.")
            m.train(); loss_sum=0.; count=0; last=time.monotonic()
            for b,(x,y) in enumerate(train_dl,1):
                x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=amp):
                    loss=nn.functional.cross_entropy(m(x),y)
                if not torch.isfinite(loss): raise FloatingPointError("Pérdida NaN/Inf.")
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                loss_sum+=float(loss.detach())*len(y); count+=len(y)
                if b==1 or b==len(train_dl) or time.monotonic()-last>=cfg["batch_log_seconds"]:
                    elapsed=time.monotonic()-begin
                    log.log(f"LOTE {b}/{len(train_dl)}, época {epoch+1}, loss={loss_sum/count:.5f}; ETA entrenamiento época={(elapsed/b*(len(train_dl)-b))/60:.1f} min.")
                    last=time.monotonic()
                if time.monotonic()>=deadline:
                    raise PauseRun("Límite de sesión: se conserva la última época completa; la época parcial se repetirá.")
            m.eval(); yy=[]; pp=[]
            log.log(f"VALIDACION época {epoch+1}: {len(val_dl)} lotes.")
            with torch.no_grad():
                for x,y in val_dl:
                    with torch.autocast(device_type=device.type,enabled=amp): o=m(x.to(device))
                    yy.extend(y.tolist()); pp.extend(o.argmax(1).cpu().tolist())
            score=metric_values(yy,pp)["macro_f1"]
            if score>best: best=score; best_state=cpu_state(m)
            history.append(dict(epoch=epoch+1,loss=loss_sum/count,val_macro_f1=score,seconds=time.monotonic()-begin))
            log.log(f"GUARDANDO checkpoint época {epoch+1}, val_macro_f1={score:.5f}, mejor={best:.5f}.")
            atomic_torch(ckpt,dict(signature=signature,epoch=epoch+1,model=cpu_state(m),
                optimizer=opt.state_dict(),scaler=scaler.state_dict(),best=best,best_model=best_state,history=history))
            atomic_csv(run_dir/"history.csv",pd.DataFrame(history))
            log.log(f"EPOCA {epoch+1} GUARDADA en {time.monotonic()-begin:.1f} s; {cfg['epochs']-epoch-1} épocas restantes en esta corrida.")
        if best_state is None: raise RuntimeError("No hay estado válido para test.")
        log.log(f"TEST {name}/{strategy}/{seed}: {len(te)} imágenes; carga mejor época por validación.")
        m.load_state_dict(best_state); m.eval(); probabilities=[]; yy=[]
        with torch.no_grad():
            for x,y in loader(df.iloc[te],transforms(cfg),cfg):
                with torch.autocast(device_type=device.type,enabled=amp): o=m(x.to(device))
                probabilities.append(o.float().softmax(1).cpu().numpy()); yy.extend(y.tolist())
        probs=np.concatenate(probabilities); preds=probs.argmax(1)
        out=pd.DataFrame(dict(image_id=df.image_id.iloc[te].to_numpy(),y_true=yy,y_pred=preds))
        for i,c in enumerate(CLASSES): out[f"prob_{c}"]=probs[:,i]
        atomic_csv(run_dir/"predictions.csv",out)
        atomic_json(run_dir/"complete.json",dict(signature=signature,domain=domain,model=name,
            strategy=strategy,seed=seed,metrics=metric_values(yy,preds),epochs=cfg["epochs"],
            predictions_sha256=file_hash(run_dir/"predictions.csv")))
        # Only last.pt in this exact completed run is obsolete; retain small outputs/history.
        ckpt.unlink(missing_ok=True)
        log.log(f"CORRIDA COMPLETA {domain}/{name}/{strategy}/{seed}; predicciones guardadas.")
        return True
    finally:
        del m,opt,scaler,best_state,train_dl,val_dl
        gc.collect()
        if device.type=="cuda": torch.cuda.empty_cache()

def cliffs_delta(a,b):
    return float(np.sign(np.asarray(a)[:,None]-np.asarray(b)[None,:]).mean())
def benjamini_hochberg(p):
    p=np.asarray(p,float); order=np.argsort(p); adjusted=np.empty(len(p))
    ranked=p[order]*len(p)/np.arange(1,len(p)+1)
    adjusted[order]=np.minimum(1,np.minimum.accumulate(ranked[::-1])[::-1])
    return adjusted
def summarize(root,cfg,signature,log):
    domain_labels=json.loads((root/"project_state.json").read_text())["scientific"]["domain_labels"]
    rows=[]; pc=[]
    for f in sorted((root/"runs").glob("*/complete.json")):
        info=json.loads(f.read_text())
        finished(f.parent,signature)
        key={k:info[k] for k in ["domain","model","strategy","seed"]}
        rows.append(dict(**key,**info["metrics"]))
        g=pd.read_csv(f.parent/"predictions.csv")
        pr,re,f1,su=precision_recall_fscore_support(g.y_true,g.y_pred,labels=[0,1,2],zero_division=0)
        for i,c in enumerate(domain_labels[info["domain"]]): pc.append(dict(**key,label=c,precision=pr[i],recall=re[i],f1=f1[i],support=int(su[i])))
    metrics=pd.DataFrame(rows,columns=["domain","model","strategy","seed","accuracy","balanced_accuracy","macro_f1"])
    atomic_csv(root/"metrics_by_run.csv",metrics)
    atomic_csv(root/"per_class_metrics.csv",pd.DataFrame(pc))
    stats=[]
    for (domain,model),g in metrics.groupby(["domain","model"]):
        for metric in ["accuracy","balanced_accuracy","macro_f1"]:
            pivot=g.pivot(index="seed",columns="strategy",values=metric)
            if not {"random","group"}.issubset(pivot.columns): continue
            pivot=pivot.dropna(); a=pivot.random.to_numpy(); b=pivot["group"].to_numpy(); n=len(a)
            if n==0: continue
            d=a-b; lo=hi=llo=lhi=float("nan"); p=float("nan")
            if n>=2:
                rng=np.random.default_rng(123); inds=rng.integers(0,n,size=(cfg["bootstrap_samples"],n))
                am=a[inds].mean(1); bm=b[inds].mean(1)
                lo,hi=np.percentile(am-bm,[2.5,97.5])
                if np.all(am>0): llo,lhi=np.percentile(100*(am-bm)/am,[2.5,97.5])
                p=1. if np.all(d==0) else exact_wilcoxon(d)
            stats.append(dict(domain=domain,model=model,metric=metric,n_pairs=n,
                expected_pairs=len(cfg["seeds"]),complete=n==len(cfg["seeds"]),
                random_mean=float(a.mean()),group_mean=float(b.mean()),difference=float(d.mean()),
                difference_ci95_low=lo,difference_ci95_high=hi,
                lsi_percent=float(100*d.mean()/a.mean()) if a.mean()>0 else float("nan"),
                lsi_ci95_low=llo,lsi_ci95_high=lhi,cliffs_delta=cliffs_delta(a,b),wilcoxon_p=p))
    st=pd.DataFrame(stats)
    if not st.empty:
        st["p_fdr_bh"]=np.nan
        # FDR is released only when the prespecified family is complete.
        domains=sorted(pd.read_csv(root/"manifest.csv").domain.unique())
        expected=len(domains)*len(cfg["models"])*3
        if len(st)==expected and st.complete.all() and st.wilcoxon_p.notna().all():
            st["p_fdr_bh"]=benjamini_hochberg(st.wilcoxon_p)
    atomic_csv(root/"stats.csv",st)
    log.log(f"RESUMEN: {len(metrics)} corridas terminadas. Resultados parciales no equivalen a estudio completo.")
    return len(metrics)

def exact_wilcoxon(d):
    from scipy.stats import rankdata
    rounded=np.round(d,12); nz=rounded[rounded!=0]
    if len(nz)==0: return 1.
    ranks=rankdata(abs(nz),method='average')
    signs=2*((np.arange(2**len(nz))[:,None] >> np.arange(len(nz))) & 1)-1
    return float((abs(signs@ranks)>=abs(np.sign(nz)@ranks)-1e-12).mean())

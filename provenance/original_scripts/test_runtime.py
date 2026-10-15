import sys,json,shutil,ast,time,tempfile
from pathlib import Path
sys.path.insert(0,str(Path('work/qa_packages').resolve()))
sys.path.insert(0,str(Path('outputs/Cierre_observaciones').resolve()))
import runtime_v5 as r
import engine_v5 as b
import torch,numpy as np,pandas as pd
from PIL import Image
torch.set_num_threads(2)
root=Path('outputs/Cierre_observaciones/estado_inicial')
state,plans,count=r.validate(root);assert count==60 and len(plans)==2
print('PASS 60 particiones, 60 resultados verificables, 180 pendientes',flush=True)
class Log:
 def log(self,msg): print(msg,flush=True)
with tempfile.TemporaryDirectory(dir='work/closure') as temp:
 t=Path(temp);copy=t/'state';shutil.copytree(root,copy)
 assert b.summarize(copy,{**b.DEFAULT,'bootstrap_samples':50},r.SIGNATURE,Log())==60
 assert set(pd.read_csv(copy/'per_class_metrics.csv').label)=={'healthy','early_blight','late_blight'}
 # Tampering and incompatible experiment must fail before any model construction.
 manifest=copy/'manifest.csv';prior=manifest.read_bytes();manifest.write_bytes(prior+b'\n')
 try:r.validate(copy);raise AssertionError('No rechazó corrupción')
 except ValueError:pass
 manifest.write_bytes(prior)
 try:r.validate(copy,'wrong');raise AssertionError('No rechazó otro protocolo')
 except ValueError:pass
 print('PASS corrupción y firma incompatible rechazadas',flush=True)
 # Synthetic only: same training routine, interruption after an epoch, restart,
 # compare predictions against an uninterrupted run and verify completed-run skip.
 rows=[]
 for i in range(36):
  path=t/f'{i}.png';Image.fromarray(np.random.default_rng(i).integers(0,256,(20,20,3),dtype=np.uint8)).save(path)
  rows.append(dict(image_id=f'fixture_{i}',path=str(path),label_id=i%3))
 df=pd.DataFrame(rows);parts=(np.arange(24),np.arange(24,30),np.arange(30,36))
 cfg={**b.DEFAULT,'smoke_test':True,'epochs':3,'image_size':16,'batch_size':6,'mixed_precision':False,'pretrained':False}
 b.CLASSES=['healthy','fungi','phytophthora']
 interrupted=t/'interrupted';whole=t/'whole'
 class Interrupt:
  def log(self,msg):
   if msg.startswith('EPOCA 1 GUARDADA'):raise b.PauseRun('interrupción técnica')
 try:b.train_one(df,parts,'synthetic','tiny','group',42,interrupted,cfg,'qa-only',torch.device('cpu'),Interrupt(),time.monotonic()+60)
 except b.PauseRun:pass
 assert (interrupted/'last.pt').exists() and not (interrupted/'complete.json').exists()
 for dest in [interrupted,whole]: b.train_one(df,parts,'synthetic','tiny','group',42,dest,cfg,'qa-only',torch.device('cpu'),Log(),time.monotonic()+60)
 assert b.file_hash(interrupted/'predictions.csv')==b.file_hash(whole/'predictions.csv')
 assert 'prob_fungi' in pd.read_csv(whole/'predictions.csv')
 oldbuild=b.build_model
 def forbid(*a,**kw):raise AssertionError('Reentrenó resultado completo')
 b.build_model=forbid
 assert b.train_one(df,parts,'synthetic','tiny','group',42,whole,cfg,'qa-only',torch.device('cpu'),Log(),time.monotonic()+60)
 b.build_model=oldbuild
 print('PASS reanudación idéntica, etiquetas de campo y omisión sin modelo',flush=True)
 assert b.exact_wilcoxon(np.ones(15))==2/32768
 assert b.exact_wilcoxon(np.zeros(15))==1
 assert b.exact_wilcoxon(np.array([1.,-1.]))==1
 r.export(copy,Log());assert (t/'results_closure_v5.zip').exists()
 print('PASS Wilcoxon exacto, resumen y exportación ZIP',flush=True)
Path('outputs/Cierre_observaciones/QA_notebook_v5.txt').write_text('PASS: validación de 60 particiones y 60 corridas reales; rechazo de corrupción y firma incompatible; entrenamiento técnico sintético de tres épocas en CPU; interrupción y reanudación con predicciones idénticas; omisión de corrida terminada sin construir modelo; etiquetas de campo; Wilcoxon exacto; resumen y exportación.\nLas pruebas sintéticas no son resultados del artículo. El entrenamiento de los cuatro modelos con los datos reales en GPU se verifica en Kaggle.\n',encoding='utf-8')

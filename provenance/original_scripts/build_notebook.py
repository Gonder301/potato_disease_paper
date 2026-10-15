import sys,json,ast,shutil
from pathlib import Path
sys.path.insert(0,str(Path('work/qa_packages').resolve()))
import nbformat
OUT=Path('outputs/Cierre_observaciones')
original=Path('work/rebuild_v42.py').read_text(encoding='utf-8')
# Preserve the actual training code; replace reporting only.
engine=original[:original.index('def export_bundle(')]
engine=engine.replace('for i,c in enumerate(CLASSES): pc.append', 'for i,c in enumerate(domain_labels[info["domain"]]): pc.append')
engine=engine.replace('    rows=[]; pc=[]\n', '    domain_labels=json.loads((root/"project_state.json").read_text())["scientific"]["domain_labels"]\n    rows=[]; pc=[]\n')
engine=engine.replace('float(wilcoxon(d,method="auto").pvalue)', 'exact_wilcoxon(d)')
engine+='''
def exact_wilcoxon(d):
    from scipy.stats import rankdata
    rounded=np.round(d,12); nz=rounded[rounded!=0]
    if len(nz)==0: return 1.
    ranks=rankdata(abs(nz),method='average')
    signs=2*((np.arange(2**len(nz))[:,None] >> np.arange(len(nz))) & 1)-1
    return float((abs(signs@ranks)>=abs(np.sign(nz)@ranks)-1e-12).mean())
'''
def function_dump(s,name):
    return ast.dump(next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name))
for name in ['train_one','build_model','transforms','loader','seed_all','metric_values']:
    assert function_dump(original,name)==function_dump(engine,name)
(OUT/'engine_v5.py').write_text(engine,encoding='utf-8')
shutil.copy2('work/closure/runtime_v5.py',OUT/'runtime_v5.py')
nb=nbformat.v4.new_notebook();md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
deps='''# Ejecutar en una sesión nueva para evitar paquetes importados de otras versiones.
import sys, subprocess, importlib.metadata, time
print("DEPENDENCIAS: comprobando versiones del experimento original...", flush=True)
required={'timm':'1.0.26','ImageHash':'4.3.2','numpy':'2.0.2','pandas':'2.3.3','scikit-learn':'1.6.1','scipy':'1.16.3'}
missing=[]
for package,version in required.items():
    try: actual=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: actual=None
    if actual!=version: missing.append(f'{package}=={version}')
if missing:
    print('INSTALANDO:',missing,flush=True)
    p=subprocess.Popen([sys.executable,'-m','pip','install',*missing])
    while p.poll() is None:
        print('DEPENDENCIAS: instalación en curso...',flush=True);time.sleep(10)
    if p.returncode: raise RuntimeError('Falló pip; revisar el log antes de entrenar.')
print('DEPENDENCIAS LISTAS. El motor verifica también PyTorch y torchvision.',flush=True)
'''
nb.cells=[md('''# Corrección de identidad y fuente — v5

Plan: 240 corridas (4 modelos × 2 conjuntos × 2 particiones × 15 semillas), 10 épocas. **60 corridas aleatorias de laboratorio se reutilizan sin alterar sus predicciones; faltan 180 nuevas.**

Laboratorio: 2.152 imágenes, 538 hojas, partición agrupada por identidad publicada de PlantVillage. Campo: 1.296 imágenes de una sola fuente, categorías Healthy/Fungi/Phytophthora; agrupamiento DINO exploratorio, sin validación humana de identidad. La referencia publicada de laboratorio no constituye una nueva revisión manual ni valida el conjunto de campo.

Conectar como datos el ZIP `Datos_y_60_corridas_reutilizables_v5.zip` extraído por Kaggle, `arjuntejaswi/plant-village` y `warcoder/potato-leaf-disease-dataset`. Mantener Internet activado y GPU. Para continuar otra sesión, conectar el último ZIP `results_closure_v5.zip` y definir RESUME_DIR si se conserva también el estado inicial.

Se comprueban firmas, imágenes, etiquetas y todas las particiones antes del entrenamiento. No se calculan embeddings en GPU. Los resultados anteriores a esta corrección se conservan como comparación histórica. Esta modificación se diseñó tras revisar v4.2; no está prerregistrada.
'''),code(deps),code('SESSION_HOURS = 8.0\nRESUME_DIR = ""  # Carpeta RESULTS_closure del último output; vacío solo si hay un único estado conectado.\n'),md('## Motor y validación\nLa celda siguiente escribe el motor incluido en este notebook. El entrenamiento conserva la implementación de v4.2; el resumen utiliza etiquetas propias de cada conjunto y Wilcoxon exacto por enumeración, con diferencias redondeadas a 12 decimales. FDR de 24 pruebas solo cuando termina la familia completa.'),code('from pathlib import Path\nPath("/kaggle/working/engine_v5.py").write_text('+repr(engine)+',encoding="utf-8")\nPath("/kaggle/working/runtime_v5.py").write_text('+repr((OUT/'runtime_v5.py').read_text(encoding='utf-8'))+',encoding="utf-8")\nprint("MOTOR v5 LISTO",flush=True)'),md('''## Ejecutar y conservar los resultados

El output informa imágenes comprobadas, inicio de corrida, lotes, época, validación, guardado y avance global. Un latido cada 60 segundos indica que el proceso responde; no demuestra avance del entrenamiento. La parada controlada ocurre a las 7 h 45 min con el presupuesto predeterminado. Puede requerir varias sesiones. Una cancelación forzada de Kaggle puede impedir la exportación; conservar el ZIP después de cada cierre controlado.

`COMPLETE` significa 240 corridas verificadas. `PAUSED` conserva lo realizado. `ERROR` explica el fallo y exporta lo disponible; no lo confunde con un estudio terminado. Los checkpoints guardan cada época completa; una época interrumpida se repite. No modifica semillas, épocas ni particiones al reanudar.
'''),code('import sys, importlib\nif "/kaggle/working" not in sys.path: sys.path.insert(0,"/kaggle/working")\nimport engine_v5, runtime_v5\nimportlib.reload(engine_v5); importlib.reload(runtime_v5)\nruntime_v5.run(session_hours=SESSION_HOURS,resume_dir=RESUME_DIR)\nfrom IPython.display import FileLink, display\ndisplay(FileLink("/kaggle/working/results_closure_v5.zip"))')]
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12.13'}}
nbformat.validate(nb)
for c in nb.cells:
    if c.cell_type=='code': compile(c.source,'notebook','exec')
path=OUT/'Correccion_identidad_y_fuente_v5.ipynb';nbformat.write(nb,path)
print('Notebook válido:',path)

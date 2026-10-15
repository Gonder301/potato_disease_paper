from pathlib import Path
import sys,json,hashlib
BASE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(BASE/'work/qa_packages'))
sys.path.insert(0,str(BASE/'work/plot_packages'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits
from PIL import Image
ROOT=Path(__file__).resolve().parent
OUT=BASE/'outputs/Revision_v5_visual/Figuras';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})

# Figure 1: source-like six-box workflow, with the corrected experimental branches.
fig,ax=plt.subplots(figsize=(9,3.5));ax.set(xlim=(0,9),ylim=(0,3.5));ax.axis('off')
boxes=[(1.5,2.65,'Potato images\nLaboratory: 2,152\nSingle-source field: 1,296'),(4.5,2.65,'Group definitions\nLab: published leaf IDs\nField: DINOv2, τ = 0.10'),(7.5,2.65,'Frozen holdouts\nRandom and grouped\nTrain / val / test: 70/15/15'),(7.5,1.1,'Four classifiers\n15 seeds per condition\n10 epochs per run'),(4.5,1.1,'240 completed runs\nSaved test predictions\nand class probabilities'),(1.5,1.1,'Balanced metrics\nLSI and uncertainty\nExposure audit')]
for x,y,label in boxes:
    ax.add_patch(FancyBboxPatch((x-1.23,y-.51),2.46,1.02,boxstyle='round,pad=.08,rounding_size=.12',linewidth=.8,facecolor='#e6e6fc',edgecolor='black'))
    ax.text(x,y,label,ha='center',va='center',fontsize=10,linespacing=1.5)
for start,end in [((2.82,2.65),(3.18,2.65)),((5.82,2.65),(6.18,2.65)),((7.5,2.03),(7.5,1.73)),((6.18,1.1),(5.82,1.1)),((3.18,1.1),(2.82,1.1))]:ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1.1))
ax.text(4.5,.19,'pHash: candidate inspection only     |     SHA256: byte-identical file exposure',ha='center',va='center',fontsize=9)
fig.tight_layout(pad=.1);fig.savefig(OUT/'Figure_1_workflow.png',dpi=300);plt.close(fig)

# Figure 2: independently fitted t-SNE panels on the exact retained image IDs.
manifest=pd.read_csv(BASE/'outputs/Estudio_v5_completo/manifest.csv')
fig,axs=plt.subplots(1,2,figsize=(9,4.2))
diagnostics=[]
for ax,domain,old,heading in zip(axs,['potato_lab_leaf_id','potato_field_single_source'],['potato_lab','potato_field'],['A  Laboratory (n = 2,152)','B  Single-source field (n = 1,296)']):
    df=manifest[manifest.domain.eq(domain)].reset_index(drop=True)
    cachepath=BASE/'work/final_results/RESULTS_v4/data'/old/'embeddings.npz'
    with np.load(cachepath,allow_pickle=False) as cache:
        ids=list(cache['image_id']);lookup={str(x):i for i,x in enumerate(ids)}
        z=cache['z'][[lookup[x] for x in df.image_id]]
    assert z.shape==(len(df),384) and np.isfinite(z).all()
    print('t-SNE starting',domain,len(df),flush=True)
    with threadpool_limits(limits=1):
        ts=TSNE(n_components=2,perplexity=30,metric='cosine',init='pca',learning_rate='auto',max_iter=1000,random_state=42,n_jobs=1)
        xy=ts.fit_transform(z)
    assert np.isfinite(xy).all()
    pd.DataFrame(dict(image_id=df.image_id,domain=domain,label=df.label,x=xy[:,0],y=xy[:,1])).to_csv(OUT/f'tsne_{domain}.csv',index=False)
    for label,color in zip((["early_blight","healthy","late_blight"] if old=='potato_lab' else ['fungi','healthy','phytophthora']),['#1f77b4','#ff7f0e','#2ca02c']):
        mask=df.label.eq(label).to_numpy();ax.scatter(xy[mask,0],xy[mask,1],s=7,alpha=.55,color=color,label=label.replace('_',' ').capitalize(),linewidths=0)
    ax.set_title(heading,fontsize=11);ax.set_xticks([]);ax.set_yticks([]);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.015),ncol=3,fontsize=8,markerscale=1.5,columnspacing=1,handletextpad=.3)
    diagnostics.append(dict(domain=domain,n=len(df),embedding_sha256=hashlib.sha256(cachepath.read_bytes()).hexdigest(),kl_divergence=float(ts.kl_divergence_),iterations=int(ts.n_iter_)))
    print('t-SNE complete',domain,flush=True)
fig.tight_layout();fig.savefig(OUT/'Figure_2_tsne.png',dpi=300);plt.close(fig)

# Figure 3: full, unaltered image arrays with verified dataset hashes.
samples=json.loads((ROOT/'sample_selection.json').read_text())
fig=plt.figure(figsize=(9,5.15))
grid=fig.add_gridspec(4,4,height_ratios=[.13,1,.13,1],hspace=.16,wspace=.05)
for row,label in enumerate(['early_blight','late_blight']):
    subset=[x for x in samples if x['label']==label]
    assert len(subset)==4 and len({x['visual_cluster'] for x in subset})==1
    header=fig.add_subplot(grid[row*2,:]);header.axis('off')
    header.text(0,.2,f'{"AB"[row]}  {label.replace("_"," ").capitalize()} — same DINOv2 cluster, two published leaf identities',fontsize=10,va='bottom')
    for col,item in enumerate(subset):
        path=ROOT/item['local_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        ax=fig.add_subplot(grid[row*2+1,col]);ax.imshow(np.asarray(Image.open(path)))
        ax.set_xticks([]);ax.set_yticks([])
        leaf=item['leaf_id'].split(':::')[-1].removesuffix('.0')
        ax.set_xlabel(f'Leaf {leaf} · image {item["panel"][-1]}',fontsize=9,labelpad=3)
        for spine in ax.spines.values():spine.set_linewidth(1.5);spine.set_edgecolor('#1f77b4' if col<2 else '#e07b20')
fig.subplots_adjust(left=.02,right=.98,top=.97,bottom=.065)
fig.savefig(OUT/'Figure_3_samples.png',dpi=300);plt.close(fig)
(OUT/'sample_provenance.json').write_text(json.dumps(samples,indent=2),encoding='utf-8')
(OUT/'figure_methods.json').write_text(json.dumps(dict(tsne=dict(perplexity=30,metric='cosine',init='pca',learning_rate='auto',max_iter=1000,random_state=42,n_jobs=1,independent_panels=True,diagnostics=diagnostics),samples=dict(selection='For each of early_blight and late_blight, first numeric DINO cluster with two sorted reference leaf IDs having at least two images; first two image IDs per selected leaf.',meaning='Illustrative examples only; not manual annotation',reference_commit='7f7ecc7e1eaca78107e3affe7cb5abd9427e139a',hashes_match_experiment=True)),indent=2),encoding='utf-8')
print('FIGURES READY',OUT,flush=True)

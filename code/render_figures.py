"""Render the four figure panels from deposited coordinates, statistics and source images.

No embeddings, clustering, t-SNE fitting or classifier training is performed here.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image

from reanalyze import REPO, file_hash, require, verify_release


def render(output):
    require(not output.exists(), 'Choose a new figure output directory.')
    verify_release()
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.set(xlim=(0, 9), ylim=(0, 3.5))
    ax.axis('off')
    boxes = [
        (1.5, 2.65, 'Potato images\nLaboratory: 2,152\nSingle-source field: 1,296'),
        (4.5, 2.65, 'Group definitions\nLab: published leaf IDs\nField: DINOv2, τ = 0.10'),
        (7.5, 2.65, 'Frozen holdouts\nRandom and grouped\nTrain / val / test: 70/15/15'),
        (7.5, 1.1, 'Four classifiers\n15 seeds per condition\n10 epochs per run'),
        (4.5, 1.1, '240 completed runs\nSaved test predictions\nand class probabilities'),
        (1.5, 1.1, 'Balanced metrics\nLSI and uncertainty\nExposure audit'),
    ]
    for x, y, label in boxes:
        ax.add_patch(FancyBboxPatch((x-1.23, y-.51), 2.46, 1.02,
                                   boxstyle='round,pad=.08,rounding_size=.12',
                                   linewidth=.8, facecolor='#e6e6fc', edgecolor='black'))
        ax.text(x, y, label, ha='center', va='center', fontsize=10, linespacing=1.5)
    for start, end in [((2.82, 2.65), (3.18, 2.65)), ((5.82, 2.65), (6.18, 2.65)),
                       ((7.5, 2.03), (7.5, 1.73)), ((6.18, 1.1), (5.82, 1.1)),
                       ((3.18, 1.1), (2.82, 1.1))]:
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', lw=1.1))
    ax.text(4.5, .19, 'pHash: candidate inspection only     |     SHA256: byte-identical file exposure',
            ha='center', va='center', fontsize=9)
    fig.tight_layout(pad=.1)
    fig.savefig(output/'Figure_1_workflow.png', dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, domain, title, classes in [
        (axes[0], 'potato_lab_leaf_id', 'A  Laboratory (n = 2,152)', ['early_blight', 'healthy', 'late_blight']),
        (axes[1], 'potato_field_single_source', 'B  Single-source field (n = 1,296)', ['fungi', 'healthy', 'phytophthora']),
    ]:
        coordinates = pd.read_csv(REPO/'figures'/f'tsne_{domain}.csv')
        require(np.isfinite(coordinates[['x', 'y']]).all().all(), 'Invalid t-SNE coordinates.')
        for label, color in zip(classes, ['#1f77b4', '#ff7f0e', '#2ca02c']):
            data = coordinates[coordinates.label.eq(label)]
            ax.scatter(data.x, data.y, s=7, alpha=.55, color=color,
                       label=label.replace('_', ' ').capitalize(), linewidths=0)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.015), ncol=3, fontsize=8,
                  markerscale=1.5, columnspacing=1, handletextpad=.3)
    fig.tight_layout()
    fig.savefig(output/'Figure_2_tsne.png', dpi=300)
    plt.close(fig)

    samples = json.loads((REPO/'figures/sample_provenance.json').read_text(encoding='utf-8'))
    fig = plt.figure(figsize=(9, 5.15))
    grid = fig.add_gridspec(4, 4, height_ratios=[.13, 1, .13, 1], hspace=.16, wspace=.05)
    for row, label in enumerate(['early_blight', 'late_blight']):
        subset = [sample for sample in samples if sample['label'] == label]
        require(len(subset) == 4, 'Expected four source images per illustrated class.')
        header = fig.add_subplot(grid[row*2, :])
        header.axis('off')
        header.text(0, .2, f'{"AB"[row]}  {label.replace("_", " ").capitalize()} — same DINOv2 cluster, two published leaf identities',
                    fontsize=10, va='bottom')
        for col, sample in enumerate(subset):
            path = REPO/'samples'/sample['local_file']
            require(file_hash(path) == sample['sha256'], f'Changed sample: {path.name}')
            ax = fig.add_subplot(grid[row*2+1, col])
            with Image.open(path) as image:
                ax.imshow(np.asarray(image))
            ax.set_xticks([])
            ax.set_yticks([])
            leaf = sample['leaf_id'].split(':::')[-1].removesuffix('.0')
            ax.set_xlabel(f'Leaf {leaf} · image {sample["panel"][-1]}', fontsize=9, labelpad=3)
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_edgecolor('#1f77b4' if col < 2 else '#e07b20')
    fig.subplots_adjust(left=.02, right=.98, top=.97, bottom=.065)
    fig.savefig(output/'Figure_3_samples.png', dpi=300)
    plt.close(fig)

    stats = pd.read_csv(REPO/'analysis/estadisticas_finales.csv')
    models = ['resnet50', 'mobilenetv3_large_100', 'tf_efficientnetv2_s', 'vit_base_patch16_224']
    names = ['ResNet50', 'MobileNetV3 Large', 'EfficientNetV2 S', 'ViT B16']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharey=True)
    for ax, domain, title in zip(axes, ['potato_lab_leaf_id', 'potato_field_single_source'],
                                ['Laboratory: leaf identity groups', 'Field: visual groups']):
        table = stats[stats.domain.eq(domain) & stats.metric.eq('balanced_accuracy')].set_index('model').loc[models]
        mean = 100*table.difference.to_numpy()
        low, high = 100*table.difference_ci95_low.to_numpy(), 100*table.difference_ci95_high.to_numpy()
        ax.errorbar(mean, np.arange(4), xerr=[mean-low, high-mean], fmt='o', capsize=4, color='#275780')
        ax.axvline(0, color='#666666', linestyle='--', linewidth=.8)
        ax.set_title(title, fontsize=10)
        ax.set_yticks(np.arange(4), names)
        ax.set_xlabel('Random − grouped balanced accuracy\n(percentage points; pointwise 95% bootstrap CI)', fontsize=9)
        ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(output/'Figure_4_split_effects.png', dpi=300)
    plt.close(fig)
    print(f'Four figures rendered from published inputs: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    render(parser.parse_args().output.resolve())

"""Recompute the published v5 tables on CPU using only NumPy and pandas.

The recorded training engine is preserved separately in engine_v5.py.
This release adapter neither imports PyTorch nor fits classifiers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SIGNATURE = '773af3d89fa82d512df8da024e151be6df88ed9a06bb4ff40797612038c6824c'
ARCHIVE_SHA256 = '501c4214e73583c628e799659b7e3a69b7aeaab38af28706e374c85ac3d71874'
KEYS = ['domain', 'model', 'strategy', 'seed']
METRICS = ['accuracy', 'balanced_accuracy', 'macro_f1']
REPO = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj):
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'), default=str)
    return hashlib.sha256(data.encode()).hexdigest()


def verify_release(repo=REPO):
    manifest = json.loads((repo / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    for rel, expected in manifest['files'].items():
        path = (repo / rel).resolve()
        require(path.is_relative_to(repo.resolve()), f'Unsafe release path: {rel}')
        require(path.is_file() and file_hash(path) == expected, f'Release file changed: {rel}')
    return len(manifest['files'])


def extract_archive(archive, destination):
    require(file_hash(archive) == ARCHIVE_SHA256, 'The archive is not the recorded final v5 archive.')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), 'Repeated archive entries.')
        require(z.testzip() is None, 'Invalid archive CRC.')
        for item in z.infolist():
            path = (destination / item.filename).resolve()
            require(path.is_relative_to(destination.resolve()) and '\\' not in item.filename,
                    'Unsafe archive path.')
            require((item.external_attr >> 16) & 0o170000 != 0o120000, 'Archive symlinks are unsupported.')
        z.extractall(destination)
    root = destination / 'RESULTS_closure'
    require(root.is_dir(), 'Missing RESULTS_closure directory.')
    return root


def scores(y, prediction):
    y, prediction = np.asarray(y, dtype=int), np.asarray(prediction, dtype=int)
    require(len(y) > 0 and len(y) == len(prediction), 'Empty or mismatched labels.')
    require(np.isin(y, [0, 1, 2]).all() and np.isin(prediction, [0, 1, 2]).all(), 'Invalid label IDs.')
    matrix = np.bincount(3 * y + prediction, minlength=9).reshape(3, 3)
    support, predicted = matrix.sum(1), matrix.sum(0)
    tp = np.diag(matrix)
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(3), where=(support + predicted) != 0)
    metrics = dict(accuracy=float(tp.sum() / support.sum()),
                   balanced_accuracy=float(recall[support > 0].mean()), macro_f1=float(f1.mean()))
    return metrics, (precision, recall, f1, support)


def compare_frame(actual, expected, keys, name):
    pd.testing.assert_frame_equal(
        actual[expected.columns].sort_values(keys).reset_index(drop=True),
        expected.sort_values(keys).reset_index(drop=True),
        check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12,
        obj=name,
    )


def validate_results(root):
    state = json.loads((root / 'project_state.json').read_text())
    require(state['version'] == '5.0.0' and state['signature'] == SIGNATURE, 'Wrong experiment signature.')
    require(canonical_hash(dict(scientific=state['scientific'], input_files=state['input_files'])) == SIGNATURE,
            'Frozen scientific specification changed.')
    for rel, expected in state['input_files'].items():
        path = (root / rel).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file() and file_hash(path) == expected,
                f'Frozen input changed: {rel}')
    sci = state['scientific']
    require(sci['expected'] == 240 and len(sci['seeds']) == 15 and len(sci['models']) == 4, 'Wrong run plan.')
    manifest = pd.read_csv(root / 'manifest.csv')
    require(manifest.image_id.is_unique, 'Repeated image IDs.')
    plans, expected_runs, rows, per_class = {}, set(), [], []
    prediction_rows = 0
    for domain, labels in sci['domain_labels'].items():
        data = manifest[manifest.domain.eq(domain)].reset_index(drop=True)
        require(data.label.equals(data.label_id.map(dict(enumerate(labels)))), f'Wrong labels: {domain}')
        require(len(data) == (2152 if domain == 'potato_lab_leaf_id' else 1296), 'Wrong image count.')
        if domain == 'potato_lab_leaf_id':
            require(data.leaf_id.notna().all() and data.leaf_id.nunique() == 538, 'Wrong leaf identities.')
        else:
            require(data.source.nunique() == 1 and data.source.iloc[0] == 'warcoder/potato-leaf-disease-dataset',
                    'Field source changed.')
        local = {}
        totals = np.bincount(data.label_id, minlength=3)
        for seed in sci['seeds']:
            for strategy in ['random', 'group']:
                split = pd.read_csv(root / 'data' / domain / f'split_{strategy}_{seed}.csv')
                require(split.image_id.tolist() == data.image_id.tolist() and
                        split.label_id.tolist() == data.label_id.tolist(), 'Split IDs or labels changed.')
                parts = tuple(np.flatnonzero(split.split.eq(k)) for k in ['train', 'val', 'test'])
                require(sorted(np.concatenate(parts).tolist()) == list(range(len(data))), 'Invalid split coverage.')
                for part, target in zip(parts, [.7, .15, .15]):
                    counts = np.bincount(data.label_id.iloc[part], minlength=3)
                    require(abs(len(part) / len(data) - target) <= .020000001 and
                            np.max(abs(counts / totals - target)) <= .030000001, 'Split outside tolerances.')
                if strategy == 'group':
                    groups = [set(data.evaluation_group.iloc[part]) for part in parts]
                    require(not any(groups[a] & groups[b] for a, b in [(0, 1), (0, 2), (1, 2)]), 'Group overlap.')
                local[(strategy, seed)] = parts
                for model in sci['models']:
                    key = dict(domain=domain, model=model, strategy=strategy, seed=seed)
                    rid = f'{domain}__{model}__{strategy}__{seed}'
                    expected_runs.add(rid)
                    directory = root / 'runs' / rid
                    info = json.loads((directory / 'complete.json').read_text())
                    require(info['signature'] == SIGNATURE and info['epochs'] == 10 and
                            all(info[k] == v for k, v in key.items()), f'Wrong completion marker: {rid}')
                    pred_file = directory / 'predictions.csv'
                    require(file_hash(pred_file) == info['predictions_sha256'], f'Prediction hash changed: {rid}')
                    pred = pd.read_csv(pred_file)
                    require(pred.image_id.tolist() == data.image_id.iloc[parts[2]].tolist() and
                            pred.y_true.tolist() == data.label_id.iloc[parts[2]].tolist(), f'Wrong test IDs: {rid}')
                    require(pred.y_true.isin([0, 1, 2]).all() and pred.y_pred.isin([0, 1, 2]).all(), 'Invalid labels.')
                    probabilities = pred[[f'prob_{label}' for label in labels]].to_numpy()
                    require(np.isfinite(probabilities).all() and (probabilities >= 0).all() and
                            (probabilities <= 1).all() and np.allclose(probabilities.sum(1), 1, atol=1e-5) and
                            np.array_equal(probabilities.argmax(1), pred.y_pred), f'Invalid probabilities: {rid}')
                    history = pd.read_csv(directory / 'history.csv')
                    require(history.epoch.tolist() == list(range(1, 11)) and
                            np.isfinite(history[['loss', 'val_macro_f1', 'seconds']].to_numpy()).all() and
                            history.loss.ge(0).all() and history.val_macro_f1.between(0, 1).all() and
                            history.seconds.ge(0).all(), f'Invalid epoch history: {rid}')
                    metrics, (precision, recall, f1, support) = scores(pred.y_true, pred.y_pred)
                    require(all(abs(value - info['metrics'][name]) <= 1e-12 for name, value in metrics.items()),
                            f'Marker metrics differ: {rid}')
                    rows.append(dict(**key, **metrics))
                    for i, label in enumerate(labels):
                        per_class.append(dict(**key, label=label, precision=precision[i], recall=recall[i],
                                              f1=f1[i], support=int(support[i])))
                    prediction_rows += len(pred)
                    if len(rows) % 40 == 0:
                        print(f'Validated {len(rows)}/240 runs', flush=True)
        plans[domain] = (data, local)
    require({p.parent.name for p in (root / 'runs').glob('*/complete.json')} == expected_runs, 'Unexpected runs.')
    require(len(rows) == 240 and prediction_rows == 62160, 'Incomplete predictions.')
    metrics, per_class = pd.DataFrame(rows), pd.DataFrame(per_class)
    compare_frame(metrics, pd.read_csv(root / 'metrics_by_run.csv'), KEYS, 'metrics_by_run')
    compare_frame(per_class, pd.read_csv(root / 'per_class_metrics.csv'), KEYS + ['label'], 'per_class_metrics')
    return metrics, per_class, plans


def bh(p):
    p = np.asarray(p, float)
    order, adjusted = np.argsort(p), np.empty(len(p))
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return adjusted


def exact_wilcoxon(difference):
    rounded = np.round(difference, 12)
    nonzero = rounded[rounded != 0]
    if len(nonzero) == 0:
        return 1.0
    ranks = pd.Series(abs(nonzero)).rank(method='average').to_numpy()
    signs = 2 * ((np.arange(2 ** len(nonzero))[:, None] >> np.arange(len(nonzero))) & 1) - 1
    return float((abs(signs @ ranks) >= abs(np.sign(nonzero) @ ranks) - 1e-12).mean())


def statistics(metrics):
    rows = []
    for (domain, model), group in metrics.groupby(['domain', 'model']):
        for metric in METRICS:
            pivot = group.pivot(index='seed', columns='strategy', values=metric)
            a, b = pivot.random.to_numpy(), pivot.group.to_numpy()
            d = a - b
            ix = np.random.default_rng(123).integers(0, 15, size=(5000, 15))
            am, bm = a[ix].mean(1), b[ix].mean(1)
            lo, hi = np.percentile(am - bm, [2.5, 97.5])
            llo, lhi = np.percentile(100 * (am - bm) / am, [2.5, 97.5])
            rows.append(dict(domain=domain, model=model, metric=metric, n_pairs=15, expected_pairs=15,
                             complete=True, random_mean=float(a.mean()), group_mean=float(b.mean()),
                             difference=float(d.mean()), difference_ci95_low=lo, difference_ci95_high=hi,
                             lsi_percent=float(100 * d.mean() / a.mean()), lsi_ci95_low=llo, lsi_ci95_high=lhi,
                             cliffs_delta=float(np.sign(a[:, None] - b[None, :]).mean()),
                             wilcoxon_p=exact_wilcoxon(d)))
    result = pd.DataFrame(rows)
    result['p_fdr_bh'] = bh(result.wilcoxon_p)
    return result


def interaction(metrics):
    rows = []
    signs = 2 * ((np.arange(2 ** 15)[:, None] >> np.arange(15)) & 1) - 1
    for domain, group in metrics.groupby('domain'):
        for metric in METRICS:
            pivot = group.pivot(index=['seed', 'model'], columns='strategy', values=metric)
            gap = (pivot.random - pivot.group).unstack('model')
            contrast = (gap.vit_base_patch16_224 - gap[[m for m in gap if m != 'vit_base_patch16_224']].mean(axis=1)).to_numpy()
            ix = np.random.default_rng(321).integers(0, 15, size=(5000, 15))
            lo, hi = np.percentile(contrast[ix].mean(1), [2.5, 97.5])
            rows.append(dict(domain=domain, metric=metric, contrast_pp=100 * contrast.mean(),
                             ci_low_pp=100 * lo, ci_high_pp=100 * hi,
                             p_signflip=float((abs(signs @ contrast / 15) >= abs(contrast.mean()) - 1e-14).mean())))
    result = pd.DataFrame(rows)
    result['q_bh_6'] = bh(result.p_signflip)
    return result


def exposure_and_allocations(plans):
    exposures, groups, splits = [], [], []
    for domain, (data, local) in plans.items():
        sizes = data.groupby('evaluation_group').size()
        table = pd.crosstab(data.evaluation_group, data.label_id)
        groups.append(dict(domain=domain, images=len(data), groups=len(sizes), largest=int(sizes.max()),
                           mean_size=sizes.mean(), singletons=int(sizes.eq(1).sum()),
                           singleton_image_percent=100 * sizes.eq(1).sum() / len(data),
                           class_purity=table.max(axis=1).sum() / len(data)))
        for (strategy, seed), parts in local.items():
            train, test = data.iloc[parts[0]], data.iloc[parts[2]]
            row = dict(domain=domain, strategy=strategy, seed=seed, test_images=len(test),
                       exact_percent=100 * test.sha256.isin(train.sha256).mean(),
                       group_percent=100 * test.evaluation_group.isin(train.evaluation_group).mean())
            if domain == 'potato_lab_leaf_id':
                row['leaf_identity_percent'] = 100 * test.leaf_id.isin(train.leaf_id).mean()
            exposures.append(row)
            for part_name, part in zip(['train', 'val', 'test'], parts):
                counts = np.bincount(data.label_id.iloc[part], minlength=3)
                splits.append(dict(domain=domain, strategy=strategy, seed=seed, part=part_name, n=len(part),
                                   class_0=int(counts[0]), class_1=int(counts[1]), class_2=int(counts[2])))
    return pd.DataFrame(exposures), pd.DataFrame(groups), pd.DataFrame(splits)


def reference_checks(repo=REPO):
    provenance = json.loads((repo / 'analysis/reference_provenance.json').read_text())
    require(file_hash(repo / 'reference/leaf-map-pinned.json') == provenance['map_sha256'], 'Leaf map hash changed.')
    reference = pd.read_csv(repo / 'reference/plantvillage_referencia_hojas.csv')
    clusters = pd.read_csv(repo / 'reference/v42_historico/potato_lab/clusters.csv')
    joined = reference[['image_id', 'leaf_id']].merge(clusters, on='image_id', validate='one_to_one')
    require(len(joined) == 2152 and joined.leaf_id.nunique() == 538, 'Reference coverage mismatch.')
    choose2 = lambda values: int(((values * (values - 1)) // 2).sum())
    positive = choose2(joined.groupby('leaf_id').size())
    rows = []
    for tau in [.1, .15, .2]:
        group = f'tau_{tau}'
        predicted = choose2(joined.groupby(group).size())
        table = pd.crosstab(joined[group], joined.leaf_id)
        tp = choose2(table.to_numpy().reshape(-1))
        rows.append(dict(tau=tau, covered_images=len(joined), leaf_ids=joined.leaf_id.nunique(),
                         predicted_positive_pairs=predicted, reference_positive_pairs=positive,
                         tp=tp, fp=predicted-tp, fn=positive-tp, pair_precision=tp/predicted,
                         pair_recall=tp/positive, identity_purity=table.max(axis=1).sum()/len(joined)))
    diagnostics = pd.DataFrame(rows)
    compare_frame(diagnostics, pd.read_csv(repo / 'analysis/validacion_grupos_identidad.csv'), ['tau'], 'identity diagnostics')
    exposures = []
    for path in sorted((repo / 'reference/v42_historico/potato_lab').glob('split_*.csv')):
        _, strategy, seed = path.stem.split('_')
        split = pd.read_csv(path).merge(reference[['image_id', 'leaf_id']], on='image_id', validate='one_to_one')
        train, test = split[split.split.eq('train')], split[split.split.eq('test')]
        covered = test.leaf_id.notna()
        exposed = int(test.loc[covered].leaf_id.isin(train.leaf_id).sum())
        exposures.append(dict(strategy=strategy, seed=int(seed), test_total=len(test),
                              test_with_reference=int(covered.sum()), exposed=exposed,
                              reference_exposure_percent=100*exposed/int(covered.sum()),
                              lower_percent_all_test=100*exposed/len(test),
                              upper_percent_all_test=100*(exposed+len(test)-int(covered.sum()))/len(test)))
    compare_frame(pd.DataFrame(exposures), pd.read_csv(repo / 'analysis/exposicion_identidad_laboratorio.csv'),
                  ['strategy', 'seed'], 'historical laboratory identity exposure')
    for sample in json.loads((repo / 'figures/sample_provenance.json').read_text(encoding='utf-8')):
        path = repo / 'samples' / sample['local_file']
        require(file_hash(path) == sample['sha256'], f'Sample bytes changed: {path.name}')
        match = reference[reference.image_id.eq(sample['image_id'])]
        require(len(match) == 1 and match.iloc[0].sha256 == sample['sha256'] and
                match.iloc[0].leaf_id == sample['leaf_id'], 'Sample reference mismatch.')
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=REPO / 'results/results_closure_v5_final.zip')
    parser.add_argument('--output', required=True, type=Path, help='A new directory; existing paths are rejected.')
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), 'Output already exists; choose a new directory.')
    count = verify_release()
    print(f'PASS: {count} release file hashes', flush=True)
    output.mkdir(parents=True)
    root = extract_archive(args.archive, output / 'inputs')
    metrics, per_class, plans = validate_results(root)
    compare_frame(metrics, pd.read_csv(REPO / 'analysis/metrics_by_run.csv'), KEYS, 'paper run metrics')
    compare_frame(per_class, pd.read_csv(REPO / 'analysis/per_class_metrics.csv'), KEYS + ['label'], 'paper per-class metrics')
    for frame, group_keys, columns, filename in [
        (metrics, KEYS[:3], METRICS, 'medias_y_desviaciones.csv'),
        (per_class, KEYS[:3] + ['label'], ['precision', 'recall', 'f1', 'support'], 'resumen_por_clase.csv'),
    ]:
        calculated = frame.groupby(group_keys)[columns].agg(['mean', 'std']).sort_index()
        recorded = pd.read_csv(REPO / 'analysis' / filename, header=[0, 1], index_col=list(range(len(group_keys))))
        pd.testing.assert_frame_equal(calculated, recorded.sort_index(), check_dtype=False,
                                      check_exact=False, atol=1e-12, rtol=1e-12, obj=filename)
    manifest = pd.read_csv(root / 'manifest.csv')
    for domain in plans:
        coordinates = pd.read_csv(REPO / 'figures' / f'tsne_{domain}.csv')
        data = manifest[manifest.domain.eq(domain)]
        require(coordinates.image_id.tolist() == data.image_id.tolist() and
                coordinates.label.tolist() == data.label.tolist() and
                np.isfinite(coordinates[['x', 'y']].to_numpy()).all(), 'Projection IDs or labels changed.')
    stats, contrasts = statistics(metrics), interaction(metrics)
    compare_frame(stats, pd.read_csv(root / 'stats.csv'), ['domain', 'model', 'metric'], 'archived statistics')
    compare_frame(stats, pd.read_csv(REPO / 'analysis/estadisticas_finales.csv'), ['domain', 'model', 'metric'], 'paper statistics')
    compare_frame(contrasts, pd.read_csv(REPO / 'analysis/interaccion_exploratoria.csv'), ['domain', 'metric'], 'interaction')
    exposure, groups, splits = exposure_and_allocations(plans)
    compare_frame(exposure, pd.read_csv(REPO / 'analysis/exposicion.csv'), ['domain', 'strategy', 'seed'], 'exposure')
    compare_frame(groups, pd.read_csv(REPO / 'analysis/grupos.csv'), ['domain'], 'groups')
    compare_frame(splits, pd.read_csv(REPO / 'analysis/particiones.csv'), ['domain', 'strategy', 'seed', 'part'], 'allocations')
    diagnostics = reference_checks()
    for name, frame in dict(metrics_by_run=metrics, per_class_metrics=per_class, stats=stats,
                            interaccion_exploratoria=contrasts, exposicion=exposure, grupos=groups,
                            particiones=splits, validacion_grupos_identidad=diagnostics).items():
        frame.to_csv(output / f'{name}.csv', index=False)
    report = dict(status='PASS', release_files_verified=count, completed=240, prediction_rows=62160,
                  histories_verified=240, splits_verified=60, primary_tests=24,
                  primary_q_below_005=int(stats.p_fdr_bh.lt(.05).sum()), interaction_tests=6,
                  interaction_q_below_005=int(contrasts.q_bh_6.lt(.05).sum()),
                  all_compared_tables_match=True, original_archive_sha256=ARCHIVE_SHA256,
                  experiment_signature=SIGNATURE, manual_field_validation='NOT_AVAILABLE',
                  used_gpu=False, trained_models=False,
                  environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__))
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

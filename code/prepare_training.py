"""Prepare a new training state using the published frozen inputs, optionally reusing 60 runs."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from reanalyze import REPO, SIGNATURE, extract_archive, file_hash, require


def prepare(archive, output, reuse=False):
    output = output.resolve()
    require(not output.exists(), 'Choose a new state directory; existing files are not overwritten.')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep temporary extraction on the selected writable volume, including sandboxes.
    with tempfile.TemporaryDirectory(prefix='.v5-frozen-', dir=output.parent) as temp:
        require(Path(temp).resolve().is_relative_to(output.parent), 'Unsafe temporary directory.')
        source = extract_archive(archive, Path(temp))
        state = json.loads((source / 'project_state.json').read_text())
        require(state['signature'] == SIGNATURE, 'Wrong experiment state.')
        # Validate all frozen inputs before creating a training directory.
        for rel, expected in state['input_files'].items():
            path = (source / rel).resolve()
            require(path.is_relative_to(source.resolve()) and file_hash(path) == expected, f'Changed input: {rel}')
        output.mkdir(parents=True)
        for rel in [*state['input_files'], 'project_state.json', 'origin_environment.json',
                    'reference_provenance.json', 'reuse_audit.csv']:
            target = output / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / rel, target)
        reused = 0
        if reuse:
            for directory in sorted((source / 'runs').glob('potato_lab_leaf_id__*__random__*')):
                shutil.copytree(directory, output / 'runs' / directory.name)
                reused += 1
            require(reused == 60, 'Expected 60 laboratory random runs.')
        print(f'Frozen v5 state prepared: {reused}/240 recorded runs retained; {240-reused} runs to train.', flush=True)
        print(f'State directory: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=REPO / 'results/results_closure_v5_final.zip')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reuse-lab-random', action='store_true', help='Reuse the original 60 laboratory random runs.')
    args = parser.parse_args()
    prepare(args.archive, args.output, args.reuse_lab_random)


if __name__ == '__main__':
    main()

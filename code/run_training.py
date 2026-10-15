"""Launch the unchanged recorded v5 runtime on an explicit frozen state and image root."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--session-hours', type=float, default=8.)
    parser.add_argument('--reserve-minutes', type=float, default=15.)
    args = parser.parse_args()
    state, images, output = args.state.resolve(), args.images.resolve(), args.output.resolve()
    if not state.is_dir() or not images.is_dir():
        raise ValueError('State and image directories must exist.')
    if output == state or output.is_relative_to(state) or state.is_relative_to(output):
        raise ValueError('Use separate state and output directories.')
    if args.session_hours <= 0 or not 0 <= args.reserve_minutes < args.session_hours * 60:
        raise ValueError('The reserve must be smaller than the positive session budget.')
    import runtime_v5
    runtime_v5.run(output_root=str(output), input_root=str(images), resume_dir=str(state),
                   session_hours=args.session_hours, reserve_minutes=args.reserve_minutes)


if __name__ == '__main__':
    main()

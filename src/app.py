import argparse
import os
import sys
import subprocess
from pathlib import Path

# Simple orchestrator: train if no model, then start server.
# Usage examples:
#   python src\app.py --data-dir data --train --epochs 10
#   python src\app.py --serve --port 8000
#   python src\app.py --train --serve --data-dir data --epochs 10 --arch efficientnet_b0


def run(cmd: list, cwd: Path | None = None):
    print("Running:", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None)
    proc.wait()
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Potato leaf disease end-to-end runner")
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--model-path', type=str, default='models/best_model.pt')
    parser.add_argument('--train', action='store_true', help='Run training step')
    parser.add_argument('--serve', action='store_true', help='Start local FastAPI server')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--arch', type=str, default='efficientnet_b0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--save-xai-samples', type=int, default=4)
    parser.add_argument('--notes', type=str, default='App orchestrated run')
    args = parser.parse_args()

    # Resolve project paths based on this file's location
    src_dir = Path(__file__).parent
    project_root = src_dir.parent
    models_dir = project_root / 'models'
    os.makedirs(models_dir, exist_ok=True)

    # Train if requested or if model is missing
    need_train = args.train or (not os.path.exists(args.model_path))
    if need_train:
        train_py = project_root / 'src' / 'train.py'
        code = run([
            sys.executable, str(train_py),
            '--data-dir', args.data_dir,
            '--arch', args.arch,
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--lr', str(args.lr),
            '--val-split', str(args.val_split),
            '--output-dir', str(models_dir),
            '--save-xai-samples', str(args.save_xai_samples),
            '--notes', args.notes
        ], cwd=project_root)
        if code != 0:
            print('Training failed. See logs above.')
            sys.exit(code)
        if not os.path.exists(args.model_path):
            # keep using models/best_model.pt default
            args.model_path = 'models/best_model.pt'

    if args.serve:
        # Start FastAPI server
        print('Starting server... Open http://127.0.0.1:%d' % args.port)
        code = run(['uvicorn', 'src.server:app', '--host', '127.0.0.1', '--port', str(args.port)], cwd=project_root)
        sys.exit(code)

    # If neither train nor serve chosen, just exit after preparing model
    print('Done. Model available at', args.model_path)


if __name__ == '__main__':
    main()

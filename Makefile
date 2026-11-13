.PHONY: setup test lint download inspect track
setup:
	python -m pip install -e '.[dev]'
test:
	python -m pytest -q
lint:
	python -m compileall -q src scripts
download:
	python scripts/download_tum.py --sequence freiburg1_xyz --destination data
inspect:
	rgbd-lab inspect data/rgbd_dataset_freiburg1_xyz --max-frames 40
track:
	rgbd-lab track data/rgbd_dataset_freiburg1_xyz --max-frames 120 --output results/fr1-xyz-first120

# Sea Turtle Detection — Project Makefile
# Usage: make <target>  (run from repo root)
#
# Windows note: use Git Bash or WSL to run make commands.
# All python commands use the project venv at .venv/

PYTHON  := .venv/Scripts/python
PIP     := .venv/Scripts/pip
SCRIPTS := scripts

# ── video source for tracking (override with: make track VIDEO=videos/myvideo.mp4) ──
VIDEO   := videos/turtle_footage.mp4
NAME    := yolov9m_bytetrack

.PHONY: help venv install check-models benchmark test-eval track clean-results

# ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Sea Turtle Detection — available commands:"
	@echo ""
	@echo "  make venv           Create .venv and install dependencies"
	@echo "  make install        Install/update dependencies into existing .venv"
	@echo "  make check-models   Validate all 16 pretrained model weights in models/"
	@echo "  make benchmark      Train all 16 models (100 epochs each, ~20 h)"
	@echo "  make test-eval      Run test-set evaluation on all 16 trained models"
	@echo "  make track          Run ByteTrack on a video"
	@echo "                        VIDEO=<path>  source video (default: $(VIDEO))"
	@echo "                        NAME=<name>   output folder name (default: $(NAME))"
	@echo "  make clean-results  Remove generated CSVs and tracking outputs (keeps plots)"
	@echo ""

# ─────────────────────────────────────────────────────────────────────
venv:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install ultralytics

install:
	$(PIP) install --upgrade pip
	$(PIP) install ultralytics

# ─────────────────────────────────────────────────────────────────────
check-models:
	$(PYTHON) $(SCRIPTS)/check_models.py

benchmark:
	$(PYTHON) $(SCRIPTS)/benchmark_yolo_models.py

test-eval:
	$(PYTHON) $(SCRIPTS)/test_evaluation.py

# ─────────────────────────────────────────────────────────────────────
track:
	$(PYTHON) $(SCRIPTS)/tracking.py \
		--source $(VIDEO) \
		--name $(NAME)

# ─────────────────────────────────────────────────────────────────────
clean-results:
	@echo "Removing generated CSVs and tracking outputs..."
	rm -f results/benchmark_results.csv
	rm -f results/test_results.csv
	rm -rf results/tracking/
	@echo "Done. Plots in results/test_runs/ and results/images/ kept."

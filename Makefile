include .env
export

SHELL := /bin/bash

act_yolov8:
	source activate env_yolov8

import_yolov8:
	conda env create -n env_yolov8 --file ./YOLOv8/env_yolov8.yml
	act_yolov8

deact_yolov8:
	conda deactivate

info:
	conda info --envs

#Local
act:
	source .venv/Scripts/activate

deact:
	deactivate


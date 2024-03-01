include ./yolov8_local/yolov8.env
export

SHELL := /bin/bash

#Local
act:
	source .venv/Scripts/activate

act_test:
	source test-env/Scripts/activate

deact:
	deactivate


# sea-turtles-detection
Detect sea turtles from drone imagery.

## Create env
I'm using an env for each notebook.

How to create a fresh env using conda:
```
conda create -n env_yolov8
``````
Activate env:
```
source activate env_yolov8
```
Install packages:
```
conda install package_name
```
Deactivate env:
```
source deactivate
```
**Export Packages:**
```
conda env export > env_yolov8.yml
```
**Import conda env from file:**
```
conda env create -n env_yolov8 --file env_yolov8.yml
```

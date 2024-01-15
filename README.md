# sea-turtles-detection
Detect sea turtles from drone imagery.

## Dataset

The dataset is available on Roboflow at this link:
<br /> [https://universe.roboflow.com/gabriel-esteves-dy2cw/sea-turtles-yia2e/dataset/1](https://universe.roboflow.com/gabriel-esteves-dy2cw/sea-turtles-yia2e/dataset/1).

The dataset contains 1,358 annotated images of turtles.

This dataset can be used to train and evaluate object detection models for turtle detection.

## Datasets Compare
| Dataset ID | Dataset Name | Train Set | Validation Set | Test Set | Data augmentation | Version | Link/Project |
|---|---|---|---|---|---|---|---|
|1|B|968|272|118|-|1|[sea-turtles-yia2e](https://universe.roboflow.com/gabriel-esteves-dy2cw/sea-turtles-yia2e)|
|2|W|15488|4352|1888|Tile|1|[seaturtletile](https://universe.roboflow.com/seaturtletile/seaturtletile/model/1)|




## Models Compare
| Model Run | Model architecture | Learning Rate | Optimizer | Training epochs | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Dataset ID | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | YOLOv8 | 0.01 | SGD | 50 | 49 | 0.748 | 0.764 | 0.809 | 0.449 | 1 | Colab/Drive |
| 2 | YOLOv8 | 0.01 | SGD | 5 | 5 | 0.652 | 0.419 | 0.453 |  0.248 | 2 | Colab/Drive |
| 3 | YOLOv8 | 0.01 | SGD | 150 | - | 0.830 | 0.92 | 0.825 |  - | 1 | train2 |
| 4 | YOLOv8 | 0.01 | SGD | 400 | 176 | 0.824 | 0.92 | 0.823 |  - | 1 | train3 |
| 5 | YOLOv8 | 0.01 | SGD | 100 | - | 0.786 | 0.91 | 0.827 |  - | 2 | train |
| 6 | YOLOv5 | 0.01 | SGD | 100 | - | - | - | - |  - | 1 | train |
| 7 | YOLOv8m | 0.01 | SGD | 100 | 55 | 0.86623 | 0.66613 | 0.79612 |  0.48666 | 1 | train3 |

## Environment File

Each model has a `.env` file stored in your Google Drive at `My_Drive/Colab_Notebooks/env`.

 You can request access to the `.env` files if needed. There is a sample `.env` file in each model folder in the repository.

To access the `.env` files, mount your Google Drive in the Colab notebook.


## Edit Colab Notebook

To edit a Colab notebook on GitHub:

1. Create a new branch on GitHub.

2. Click the **Open in Colab** link at the top of the notebook. 
<br /> If the link is deprecated, choose the notebook from this link:
<br /> https://colab.research.google.com/github/gab-es21/sea-turtles-detection/.<br /> Make sure to select the notebook on the newly created branch.

3. To edit the notebook, you need to create a copy on your Google Drive.
<br /> Click **File > Save a copy in Drive**.

4. Make your edits to the notebook.

5. To save your changes back to GitHub.
<br /> Click **File > Save a copy in GitHub**.
Save it on the main branch.

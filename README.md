# Pedestrian Detection

## Problem Statement

Pedestrian detection is a crucial aspect of self-driving cars - all detectors should be highly precise for its application to be safe. While there are a large variety of datasets for this task, most state-of-the-art detectors performance is highly tailored to their train and target datasets. Such models perform poorly in the real world where there could be a variety of different situations and their challenges.

Pedestrian detection as a task faces a wide variety of challenges, not limited to:
-	Different environmental situations - rain, snow, sunny, foggy, gloomy
-	Pedestrians occluded by other pedestrians and road side objects
-	Varying density of pedestrians based on locations - crowded cities have a large number of pedestrians at a time, while the density reduces significantly in smaller towns.
-	The number of unique pedestrians are limited in datasets available, further hampering generalization capabilities


Our goal is to implement the task of pedestrian detection in autonomous driving through state of the art CNN architectures, comparing their performance and analyzing the generalization ability through cross dataset evaluations.

## Description of the Repository

This repository consists of the datataset annotations, tools and configs folders along with the cloned mmdet repository.

1) Configs: Stores the .py config files to run Cascade-RCNN and CSP models on both CityPersons and EuroCityPersons datasets.

2) Datasets: Stores the respective .json files to read CityPersons and EuroCityPersons train, validation and test datasets.

3) MMDet: Stores the core model code for Cascade-RCNN and CSP models (among other models) and other supporting files.

4) Tools: Stores the code files to test and validate CityPersons and EuroCityPersons datasets.

## Usage

Kindly refer to the following sequence of installations and set up commands before running the files on command line.

```
conda create -n open-mmlab python=3.7 -y

conda activate open-mmlab

conda install cython

git clone <INSERT_OUR_REPO>

cd <REPO_NAME>

pip install torch==1.4.0 torchvision==0.5.0

pip install scipy

python setup.py develop

pip install mmcv==0.2.14
```

Prepare the CityPersons dataset from CityScapes using the following steps.

```
git clone https://github.com/mcordts/cityscapesScripts.git

pip install cityscapesscripts

python cityscapesScripts/cityscapesscripts/download/downloader.py
(Enter login credentials : username and password)

python cityscapesScripts/cityscapesscripts/download/downloader.py leftImg8bit_trainvaltest.zip

python -m zipfile -e leftImg8bit_trainvaltest.zip /datsets/CityPersons/leftImg8bit_trainvaltest/

cd /datsets/CityPersons/leftImg8bit_trainvaltest/leftImg8bit

find val -type f -print0 | xargs -0 mv -t val_all_in_folder
```

Prepare the EuroCityPersons dataset using the following steps.

```
cd ..
cd ..
cd ..

wget --auth-no-challenge --user=<enter_username> --password=<enter_password> --output-document=ECP_day_img_train.zip http://eurocity-dataset.tudelft.nl//eval/downloadFiles/downloadFile/detection?file=ecpdata%2Fecpdataset_v1%2FECP_day_img_train.zip

wget --auth-no-challenge --user=<enter_username> --password=<enter_password> --output-document=ECP_day_img_val.zip http://eurocity-dataset.tudelft.nl//eval/downloadFiles/downloadFile/detection?file=ecpdata%2Fecpdataset_v1%2FECP_day_img_val.zip

wget --auth-no-challenge --user=<enter_username> --password=<enter_password> --output-document=ECP_day_labels_val.zip http://eurocity-dataset.tudelft.nl//eval/downloadFiles/downloadFile/detection?file=ecpdata%2Fecpdataset_v1%2FECP_day_labels_val.zip

python -m zipfile -e ECP_day_img_train.zip /datasets/EuroCity/

python -m zipfile -e ECP_day_img_val.zip /datasets/EuroCity/

python -m zipfile -e ECP_day_labels_val.zip /datasets/EuroCity/

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.6.0/index.html
```

### Model Training

Training on CityPersons using Cascade-RCNN with the following code line:

```
python tools/train.py configs/cityperson/cascade_hrnet.py
```

Alternately, training on CityPersons using CSP model with the following code line:

```
python tools/train.py configs/cityperson/csp_r50.py
```

### Model Testing

Testing model with the weights obtained after training on CityPersons.

1) Testing Cascade-RCNN model on CityPersons dataset

```
python tools/test_city_person.py configs/cityperson/cascade_hrnet.py work_dirs/cityperson_cascade_rcnn_hrnetv2p_w32/epoch_ 20 21 --out result_citypersons_cascade.json --mean_teacher
```

2) Testing Cascade-RCNN model on EuroCityPersons dataset

```
python tools/test_euroCity.py configs/eurocity/cascade_hrnet.py work_dirs/eurocity_cascade_rcnn_hrnetv2p_w32/epoch_ 20 21 --out result_eurocity_cascade.json --mean_teacher
```
3) Testing CSP model on CityPersons dataset

```
python tools/test_city_person.py configs/cityperson/csp_r50.py work_dirs/cityperson_csp_r50v2p_w32/epoch_ 20 21 --out result_citypersons_csp.json --mean_teacher
```

4) Testing CSP model on EuroCityPersons dataset

```
python tools/test_euroCity.py configs/eurocity/csp_r50.py work_dirs/eurocity_csp_r50v2p_w32/epoch_ 20 21 --out result_eurocity_csp.json --mean_teacher
```

## Results

Finally, the model performance comparison is apparent from the table below:

| Model        | Test Dataset     | Reasonable | Small  | Heavy  | All    |
| ------------ | ---------------- | ---------- | ------ | ------ | ------ |
| Cascade RCNN | CityScapes       | 15.15%     | 21.96% | 42.49% | 39.11% |
| CSP          | CityScapes       | 18.39%     | 27.21% | 49.14% | 45.29% |
| Cascade RCNN | EuroCity Persons | 20.31%     | 42.40% | 49.60% | 38.68% |
| CSP          | EuroCity Persons | 19.60%     | 52.35% | 58.41% | 49.83% |

### Training curves

The training loss and accuracy curves can be observed as below.

|              | Loss       | Accuracy |
| CSP          | ![A](https://github.com/vaibhavbagri/Pedestrian-Detection-Practical-DL/blob/main/Results/cp_csp_loss.png) | ![B](https://github.com/vaibhavbagri/Pedestrian-Detection-Practical-DL/blob/main/Results/cp_csp_acc.png) |
| Cascade RCNN | ![C](https://github.com/vaibhavbagri/Pedestrian-Detection-Practical-DL/blob/main/Results/cp_cascade_loss.png) | ![D](https://github.com/vaibhavbagri/Pedestrian-Detection-Practical-DL/blob/main/Results/cp_cascade_acc.png) |


### Sample Observations

Cascade-RCNN on CityPersons Dataset
![Cascade RCNN on CityPersons](https://github.com/vaibhavbagri/Pedestrian-Detection-Practical-DL/blob/main/Results/cp_cascade_test%20(1).png)

CSP on CityPersons Dataset
![CSP on CityPersons](https://github.com/vaibhavbagri/Pedestrian-Detection-Practical-DL/blob/main/Results/cp_csp_test%20(1).png)

Cascade-RCNN on EuroCityPersons Dataset
![Cascade RCNN on EuroCity](https://github.com/vaibhavbagri/Pedestrian-Detection-Practical-DL/blob/main/Results/ecp_cascade_test%20(1).png)

CSP on EuroCityPersons Dataset
![CSP on EuroCity](https://github.com/vaibhavbagri/Pedestrian-Detection-Practical-DL/blob/main/Results/ecp_csp_test%20(1).png)



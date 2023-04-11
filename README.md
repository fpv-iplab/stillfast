# StillFast: An End-to-End Approach for Short-Term Object Interaction Anticipation

This is the official github repository of the following publication:

F. Ragusa, G. M. Farinella, A. Furnari. StillFast: An End-to-End Approach for Short-Term Object Interaction Anticipation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. 2023.

[project web page](https://iplab.dmi.unict.it/stillfast/) | [paper](https://arxiv.org/abs/2304.03959)


## Citing StillFast Paper
If you find our work useful in your research, please use the following BibTeX entry for citation.
```
 @InProceedings{ragusa2023stillfast,
 author={Francesco Ragusa and Giovanni Maria Farinella and Antonino Furnari},
 title={StillFast: An End-to-End Approach for Short-Term Object Interaction Anticipation}, 
 booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
 year      = {2023}
 }

```

## Installation
### Requirements

#### Anaconda
An Anaconda environment with the requirements is provided in `environment.yml`. If you are using Anaconda, you can create a suitable environment with:

`conda env create -f environment.yml`

Then, activate the environment:

`conda activate stillfast`

#### Pip
We provide a list of libraries in requirements.txt. You can easy install these libraries using pip:

`pip install -r requirements.txt`

### Wandb
Wandb is enabled by default. To use it set the credentials in `wandb/settings`:

```
entity = yournickname
project = yourprojectname
base_url = https://api.wandb.ai
```
Then, login with `wandb login`.


## Model Zoo and Baselines
We provided pretrained models on EGO4D `v1` and `v2`:


| pretraining | Still | Fast | model |  config  |
| ------------- | -------------| ------------- | ------------- | ------------- | 
| EGO4D v1 | ResNet R50 | X3D_M |  [`link`](https://iplab.dmi.unict.it/sharing/StillFast/models/StillFast_EGO4D_v1.ckpt) | configs/sta/STILL_FAST_R50_X3DM_EGO4D_v1.yaml |
| EGO4D v2 | ResNet R50 | X3D_M | [`link`](https://iplab.dmi.unict.it/sharing/StillFast/models/StillFast_EGO4D_v2.ckpt) | configs/sta/STILL_FAST_R50_X3DM_EGO4D_v2.yaml |


## EGO4D Dataset
To train/test the model on the EGO4D dataset, follow the instructions provided here to download the dataset and its annotations for the Short-Term Object Interaction Anticipation task:

`https://github.com/EGO4D/forecasting/blob/main/SHORT_TERM_ANTICIPATION.md`


## Training

To train StillFast on the EGO4D dataset, execute the following command:

`python main.py --cfg configs/sta/STILLFAST_R50_X3DM_EGO4d-V2.yaml --train --exp unique_experiment_name`

Outputs will be logged to wandb and stored under the folder `output/sta/StillFast_unique_experiment_name/version_0/`

If you repeat the command, experiments will be saved under the `version_1` subdirectory and so on.

## Validation
Trained models can be validated using the following command:

`python main.py --val --test_dir output/sta/StillFast_unique_experiment_name/version_x/`

where `x` is the version number of your experiment.
After the validation phase, predictions will be saved in a json file under:

`output/sta/StillFast_unique_experiment_name/version_x/results/val.json`

Results will be printed, but you may obtain the final ones using the official [`evaluate_short_term_anticipation_results.py` script](https://github.com/EGO4D/forecasting/blob/main/SHORT_TERM_ANTICIPATION.md#evaluating-the-results).

You can evaluate the results with the following command:   

`python /path/to/forecasting/tools/short_term_anticipation/evaluate_short_term_anticipation_results.py output/sta/StillFast_unique_experiment_name/version_x/results/val.json /path/to/ego4d/annotations/fho_sta_val.json`

## Test

The `main.py` program also allows to run the model on the EGO4D test set and produce a json file to be sent to the [`leaderboard`](https://eval.ai/web/challenges/challenge-page/1623/leaderboard/3910). To test models, you can use the following commands:

`python main.py --test --test_dir output/sta/StillFast_unique_experiment_name/version_x/`

After the test phase, predictions will be saved in a json file under:

`output/sta/StillFast_unique_experiment_name/version_x/results/test.json`

To obtain results, submit the `test.json` file to the [`EGO4D Short Term Object Interaction Anticipation Challenge page`](https://eval.ai/web/challenges/challenge-page/1623/overview).

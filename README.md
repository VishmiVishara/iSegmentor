## iSegmentor
iSegmentor is a novel framework for semantic segmentation using Neural Architecture Search (NAS) and Generative Adversarial Networks (GANs).

![iSegmentor_Architecture](imgs/iSegmentor_Architecture.png)


## Requirements

+ asgiref
+ autopep8
+ django==2.2.10
+ pycodestyle
+ pytz
+ sqlparse
+ Unipath
+ dj-database-url
+ python-decouple
+ gunicorn
+ whitenoise
+ torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
+ tqdm
+ numpy
+ pydicom
+ SimpleITK
+ Pillow
+ PyYAML
+ plotly
+ opencv-python
+ pygraphviz
+ graphviz
+ scipy
+ scikit-image
+ adabound
+ tensorboardX
+ schedule


## Features:

- [x] Multi GPU support for Senerator Achitecture Search
- [x] Multi GPU support for GAN based Segmentation Model Training
- [x] Segmengation Model Evaluation and Metrics Visualization Through GUI
- [x] Segemntation Reuslts Download and Metrics Download
- [x] Dataset, Search, Train Configuration Change through GUI
- [x] Dataset Upload or Sample Dataset is Available to Experiment
- [x] Real-time Metrics Visualization
- [x] Tensorboard Monitoring Integrated to GUI




## Usage

```bash
pip3 install -r requirements.txt
```

## Citation

If you are using this repo please cite these papers
```
@INPROCEEDINGS{9605889,
  author={Ganepola, Vayangi Vishmi Vishara and Wirasingha, Torin},
  booktitle={2021 10th International Conference on Information and Automation for Sustainability (ICIAfS)}, 
  title={A Novel Framework for Semantic Image Segmentation using Generative Adversarial Networks and Neural Architecture Search}, 
  year={2021},
  volume={},
  number={},
  pages={252-257},
  doi={10.1109/ICIAfS52090.2021.9605889}}
```
```
@INPROCEEDINGS{9515223,
  author={Ganepola, Vayangi Vishmi Vishara and Wirasingha, Torin},
  booktitle={2021 IEEE 4th International Conference on Big Data and Artificial Intelligence (BDAI)}, 
  title={Generative Adversarial Networks Using Neural Architecture Search for Semantic Image Segmentation}, 
  year={2021},
  volume={},
  number={},
  pages={236-241},
  doi={10.1109/BDAI52447.2021.9515223}}
```
```
@INPROCEEDINGS{9396991,
  author={Ganepola, Vayangi Vishmi Vishara and Wirasingha, Torin},
  booktitle={2021 International Conference on Emerging Smart Computing and Informatics (ESCI)}, 
  title={Automating Generative Adversarial Networks using Neural Architecture Search: A Review}, 
  year={2021},
  volume={},
  number={},
  pages={577-582},
  doi={10.1109/ESCI50559.2021.9396991}}
```
```
@ARTICLE{8681706, 
author={Y. {Weng} and T. {Zhou} and Y. {Li} and X. {Qiu}}, 
journal={IEEE Access}, 
title={NAS-Unet: Neural Architecture Search for Medical Image Segmentation}, 
year={2019}, 
volume={7}, 
number={}, 
pages={44247-44257}, 
keywords={Computer architecture;Image segmentation;Magnetic resonance imaging;Medical diagnostic imaging;Task analysis;Microprocessors;Medical image segmentation;convolutional neural architecture search;deep learning}, 
doi={10.1109/ACCESS.2019.2908991}, 
ISSN={2169-3536}, 
month={},}
```

## References
```
@ARTICLE{8681706, 
author={Y. {Weng} and T. {Zhou} and Y. {Li} and X. {Qiu}}, 
journal={IEEE Access}, 
title={NAS-Unet: Neural Architecture Search for Medical Image Segmentation}, 
year={2019}, 
volume={7}, 
number={}, 
pages={44247-44257}, 
keywords={Computer architecture;Image segmentation;Magnetic resonance imaging;Medical diagnostic imaging;Task analysis;Microprocessors;Medical image segmentation;convolutional neural architecture search;deep learning}, 
doi={10.1109/ACCESS.2019.2908991}, 
ISSN={2169-3536}, 
month={},}
```




# People detection

Wondering how crowded is your favourite beach of Gipuzkoa (Basque Country)?

![Prediction example with ResNet 1024x1024 pretrained model](./resnet_1024x1024.jpg)

## Model

Based on [`pix2seq`](https://github.com/google-research/pix2seq), currently using ResNet-50 pretrained model, COCO object detection fine-tuned checkpoints. 

## Accuracy

Despite of being quite accurate out-of-the-box model, it fails sometimes. This is because the  camera is too far from the people. This could be handled fine-tuning the model, but it would be a time consuming task... As a baseline, it is okay.

# Installation

Clone the repository and the submodules with 
```shell
$ git clone --recurse-submodules https://github.com/r3v1/crowded_beach.git
```

then fetch the checkpoints from the LFS

```shell
$ git lfs pull
```

and finally, install the requirements (recommended using a virtual environment):
```shell
$ pip install -r requirements.txt
```

# Running

```shell
$ python src/predict.py --help                                                                                             [git][crowded_beach/.][main]
usage: predict.py [-h] -b {Zurriola,Kontxa} [-t THRESHOLD] [-s] [-d DELAY] [-m MODEL_DIR] [--from-gcloud]

Wondering how crowded is your favourite beach of Gipuzkoa (Basque Country)?

options:
  -h, --help            show this help message and exit
  -b {Zurriola,Kontxa}, --beach {Zurriola,Kontxa}
                        Beach to analyze
  -t THRESHOLD, --threshold THRESHOLD
                        Minimum score threshold
  -s, --save-prediction
                        Saves prediction JPG to last.jpg
  -d DELAY, --delay DELAY
                        Delay between frame captures. 0 disables (predict and exit)
  -m MODEL_DIR, --model-dir MODEL_DIR
                        Path to object detection model (default: /home/david/git/cameras/coco_det_finetune/resnet_640x640)
  --from-gcloud         Uses GCloud stored model
```

# Similar projects

- [Surfer Counter](https://surfercounter.com/)
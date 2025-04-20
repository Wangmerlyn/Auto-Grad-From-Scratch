# Auto-Grad Demo

This is a demo showcasing how to build an auto-gradient calculation system using the backward pass and predefined forward and backward functions.

The code style may be a bit rough, so any pull requests or issues are welcome.

This repo is accompanied by a video introducing how an auto-gradient system works. You can find my video here: [Bilibili (Chinese)](link).  
Since this repo is mostly for educational purposes, it cannot be used as a fully functioning auto-grad library. As you can see, many essential functions for auto-grad are not included.

The idea for this small project and video was inspired by [joelgrus/autograd](https://github.com/joelgrus/autograd)(he is a genius btw), with a lot of code directly taken from this repo. You can check out his [coding livestream](https://youtu.be/RxmBukb-Om4?si=BYfA1XF8M-FWkhw1), where he coded this repo live.

## Installing the Environment
If your `conda.sh` is located at `source /opt/conda/etc/profile.d/conda.sh`:
```bash
bash install_env.sh
```
or just create any virtual env you like and use 
```bash 
pip install numpy
```

## Run a test case
```bash 
python main.py
```
This test case implements a very simple grad calculation in linear transformation, and use a very simple numerical way to check our gradient is correct.
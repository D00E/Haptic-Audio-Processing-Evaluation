# Project Title

Overview of audio evaluation

## Requirements
- Ubuntu       22.04.5 LTS
- conda        24.1.2

### GPU Usage of Tensor Flow
Check the following website for which tf version to use
Align  your cuDNN and CUDA with this

https://www.tensorflow.org/install/source#gpu

This env uses 
- tensorflow                2.19.0
- nvidia-cudnn-cu12         8.9.2.26                 pypi_0    pypi
- 

- GPU Optional for faster tf audio processing
- TOrch reverts back to cpu with this environment

## Running instructions
The required conda environment file is named : audio_eval_env.yaml

1. Clone eval
2. Create and activate the environment:
   ```bash
   conda env create --file audio_eval_env.yml

   conda activate audio_eval_env
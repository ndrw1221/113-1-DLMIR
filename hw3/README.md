# DeepMIR HW3: Symbolic Generation

This project involves training a GPT-2 model to generate MIDI files in the REMI (REpresentation of Music Information) format. The generated symbolic music can be further converted into audio using MIDI synthesizers.


## Report

[Link to Report](https://docs.google.com/presentation/d/1pPbbONKcO7kG6GgJnbW-ZygljKRAxrxLuPXohke6yWc/edit?usp=sharing)

## Inference Guide

Follow the steps below to set up the environment, generate MIDI files, and convert them to audio.

### 1. Environment Setup

Make sure you have Anaconda installed before proceeding.

```bash
# Create a conda environment using the provided environment.yml file
conda env create -f environment.yml -n YOUR_ENV_NAME

# Activate the newly created environment
conda activate YOUR_ENV_NAME
```

### 2. Generation

To generate a MIDI file using the trained model, run the following script:

```bash
python generation.py
```
**Notes:**

- The generation.py script uses the trained GPT-2 model to generate music in the REMI format.
- Ensure that the trained model weights are correctly placed in the checkpoints/ directory before running the script.

### 3. Synthesize audio from MIDI

**Step 1: Install FluidSynth**
```bash
sudo apt update
sudo apt install fluidsynth
```
**Step 2: Convert MIDI to Audio**
```bash
fluidsynth -ni soundfonts/MuseScore_General.sf2 path/to/midi.mid -F path/to/output/audio/.wav
```

## Directory Structure

```
.
├── checkpoints/         # Pre-trained model weights        
├── environment.yml      # Conda environment setup file
├── generation.py        # Script for MIDI generation
└── train.py             # Script for training the model
```
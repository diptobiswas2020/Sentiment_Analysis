# Sentiment_Analysis
This repository is specially for Enhancing Sentiment Analysis Approach Using Efficient Data Preparation Strategy, Proficient Word Sense Disambiguation, and Improved Bag-of-Words Model along with Selectional Preferences based Word Sense Disambiguation

# IMDB Movie Reviews Dataset
IMDB introduces a vast amount of opinions of users on movies. The IMDB dataset is labelled and considered opinions of users are polarized. 25,000 users are engaged to opinionate 50,000 polarized opinions, and opinions are scored between [4,6] for negative polarized opinions and [7,10] for positive polarized opinions. Each movie has 30 opinions particularly or less than 30 opinions with some superfluous unlabelled data.

# Experimental Setup
This research has implemented Enhancing the Performance of Sentiment Analysis with Automatic Selectional Preference-Based Word Sense Disambiguation. This experiment has been conducted by using JupyterLab which is the latest web-based interactive development environment for notebooks, code, and data. The configuration for such detection and computation was 64GB Random Access Memory, 8GB GDDR5, GPU NVIDIA Quadro P4000, Windows 10, CPU Intel(R) Xeon(R) E5-1620 v4 @ 3.5 GHz based workstation.

# Dataset
This research utilizes IMDB movie review dataset. The dataset is available as an open source data which is completely free for all. This research collects the IMDB movie review dataset from kaggle open source platform. [Training Dataset](https://drive.google.com/file/d/13y_YcF_MUiMcxY9dYahjYriusPdejr3R/view?usp=sharing) along with [Test Dataset](https://drive.google.com/file/d/1prYmMPPFWavVmxwfMCmB2_Iq-Jhm3aqu/view?usp=sharing)

# Run Method in Windows.
If you find any mistakes or disagree with any of the explanations, please do not hesitate to submit an issue. I welcome any feedback, positive or negative!

Install the required dependencies with: 
```
py -m pip install library_name.
```

The data is organized in a ```.csv``` file as follows:
```text  |  label  |```  

- The sentiments are either positive, neutral or negative, denoted as pos, neu and neg. 
- The texts have been labeled by human annotators, where the majority label has been chosen as label.
- It is highly recommended to create a new Virtual Environment first before running the above commands.

# Instruction for Virtual Environment
- The ```venv``` module supports creating lightweight ```“virtual environments”```, each with their own independent set of Python packages installed in this implementation. 
```
python -m venv /path/to/new/virtual/environment
```
Running the above command creates the target directory (creating any parent directories that don’t exist already) and places a ```pyvenv.cfg``` file in it with a home key pointing to the Python installation from which the command was run (a common name for the target directory is ```.venv```). It also creates a ```bin``` (or Scripts on Windows) subdirectory containing a copy of the Python ```binary/binaries``` (as appropriate for the platform or arguments used at environment creation time). It also creates an (initially empty) ```lib/pythonX.Y/site-packages``` subdirectory (on Windows, this is ```Lib\site-packages```). If an existing directory is specified, it will be re-used.

# Step wise declaration.
- contain data at D folder.
- create a folder named input in the D folder.
- Type code in jupyter notebook named with ```IMDB_EDPS.py``` and ```IMDB_PWSD.py```.
- Contain ```IMDB_model.h5```, ```IMDB_dictionary.txt``` and ```IMDB_Sentiment_Labels.txt``` in the same directory.
- All the required files are available [here](https://drive.google.com/drive/folders/1mmcqWI7JLx9Q0u-6z4_3hx8Q4v9gPMOO?usp=sharing)

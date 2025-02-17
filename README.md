# Master's Thesis Project

## Overview

This repository contains the code, data, and documents for the master's thesis project conducted at the University of Glasgow. The research focuses on implementing and evaluating non-parametric change point detection algorithms. The project includes various methodologies such as CUSUM, GLR, Q-Q distance, and k-NN based detection, each designed to detect changes in high-dimensional time series data.

## Directory Structure

- `/src`: Source code for the project, organized as follows:
  - `/final`: Final versions of the source code used for the thesis.
  - `/temp`: Temporary scripts and code used during development and experimentation.
- `/data`: Datasets used in the project.
  - `/raw`: Raw datasets as received or downloaded.
  - `/processed`: Preprocessed datasets ready for analysis.
- `/results`: Experimental results, including logs, generated figures, and summary tables.
  - `/figures`: Plots and visualizations generated from experiments.
  - `/tables`: Tables summarizing experimental results.
  - `/logs`: Logs generated during the execution of experiments.
- `/report`: Documents related to the thesis, including drafts and final versions.
  - `/drafts`: Draft versions and notes.
  - `/final`: Final versions of the thesis and any related publications.
- `/notebooks`: Jupyter notebooks used for running experiments and performing data analysis.
  - `/experiments`: Notebooks used for running and documenting experiments.
  - `/analysis`: Notebooks used for data analysis and result interpretation.

## Setup & Usage

### Setting up the Environment

To set up the environment, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/username/MSc-project.git
   cd MSc-project
   ```

2. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code

1. **Prepare the data:**

   - Ensure the datasets are placed in the `/data/raw` directory.
   - Run any preprocessing scripts to generate the necessary processed data in `/data/processed`.

2. **Execute the main analysis script:**

   ```bash
   python src/final/main.py
   ```

   This will run the full set of experiments and save the results in the `/results` directory.

3. **Explore the results:**
   - Generated plots can be found in `/results/figures`.
   - Summary tables and metrics are available in `/results/tables`.

### Running Experiments in Jupyter Notebooks

1. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
2. **Navigate to `/notebooks/experiments` or `/notebooks/analysis`** to run the experiments or analyze the results.

## Dependencies

The project uses Python 3.11+ and the following Python libraries:


- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For creating static, animated, and interactive visualizations.
- **SciPy**: For scientific computing and technical computing.

Temp file



https://github.com/user-attachments/assets/1df7efcb-e9a4-4a83-b799-f54ca263f425



https://github.com/user-attachments/assets/ac361a4c-b699-4a71-8fe8-dd6193d65f0f



https://github.com/user-attachments/assets/0fe15ecc-f6f2-4fba-8646-1ed7082482e6


https://github.com/user-attachments/assets/92c9973a-722c-4500-83bf-00af9b73bfe2




https://github.com/user-attachments/assets/6e3a891e-905e-4585-8a44-04e7b13e65f6



https://github.com/user-attachments/assets/413da20f-a469-4cb4-8b54-82e92570ea67



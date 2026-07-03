# CardioLens

This repository contains the Python code developed for the Bachelor thesis titled "Development of a Web Application for Physiological Parameter Monitoring Using Smartphone-Based Photoplethysmography".

The project implements a smartphone-based pipeline to extract HR, SpO2 and BR from a PPG signal acquired from a smartphone camera


## Project Structure

├── extraction #folder containing the code needed to extract the PPG signal
├── parameters #folder with algorithms that compute the parameters
├── requirements.txt    # project dependencies
└── README.md

---

## Installation

(Optional but recommended) Create a virtual environment:

python -m venv venv

Activate it:

- macOS/Linux:
  source venv/bin/activate

- Windows:
  venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

---

## Usage

code inside extraction folder should be run first


## Requirements

The project requires:

- Python 3.11
- cv2
- os
- glob
- numpy
- pywt
- warnings
- matplotlib.pyplot
- matplotlib
- pandas as pd
- scipy
- scipy.signal
- copy
- scikit-learn
- joblib


Alternatively, install everything via:

pip install -r requirements.txt

---

## Authors

Gioele Longhi, Ludovico Morando, Luca Mercuri, Gabriele Mammarella  
University: [Politecnico di Milano]  
Degree Program: [Ingegneria Biomedica]  

---

## License

This project is released under the MIT license.

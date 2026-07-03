# CardioLens

This repository contains the Python code developed for the Bachelor thesis titled "Development of a Web Application for Physiological Parameter Monitoring Using Smartphone-Based Photoplethysmography".

The project implements a smartphone-based pipeline to extract HR, SpO2 and BR from a PPG signal acquired from a smartphone camera


## Project Structure


---
├── extraction/      # folder containing the code needed to extract the PPG signal<br>
├── parameters/      # folder with algorithms that compute the parameters<br>
├── requirements.txt # project dependencies<br>
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
In order to estimate HR and SpO2, you should first run PIPELINE RAW PPG THESIS and then the corresponding code in the "extraction folder"
For BR estimation you can run BR_EMD_like directly or PPG_to_BR first and BR_zero_crossing_final later

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

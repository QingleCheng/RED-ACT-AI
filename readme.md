# Building Seismic Response Prediction

## Overview
This project is designed for predicting the seismic response of building clusters. It leverages deep learning techniques to analyze earthquake damage patterns. The related research paper is currently under review.

## Methodology
- A deep learning-based approach is used to establish a seismic response database of building clusters.
- Based on the database, a predictive model is constructed for estimating the seismic damage levels of building clusters.
Below is the framework of the methodology used in this project:

![Methodology Framework](img/framework.png)
## Quick Start

### Prerequisites
To get started, ensure you have the required Python environment configured. Refer to the `Prediction.py` script for details on the necessary dependencies.

### Prerequisites
To run this project, you need Python installed along with the following libraries (with suggested versions):

- torch >= 2.4.1+cu124
- numpy >= 1.26.4
- pandas >= 1.3.0
- scikit-learn
- matplotlib
- joblib

You can install these dependencies using the following command:
```bash 
pip install  torch==2.4.1+cu124 numpy==1.26.4 pandas==1.5.3 scikit-learn matplotlib joblib
```
Refer to the Prediction.py script for further details on required dependencies.


### Pre-trained Models
The `model` folder contains pre-trained AI models for predicting three key metrics:
- **MIDR**: Maximum Inter-story Drift Ratio.
- **MPFA**: Maximum Peak Floor Acceleration.
- **PTFA**: Peak Top Floor Acceleration.

### Input Data
- **`building_data.xlsx`**: A sample input file used for predictions. It contains data from 47 buildings with measured earthquake responses.The dataset is sourced from the **CESMD Strong Motion Database**, and we express our gratitude for their data support.
- **`example_input_data.csv`**: Provides the basic attributes of the 47 buildings, along with seismic response information.

### Prediction Script
The `Prediction.py` script demonstrates how to use the provided models to make predictions. The script:
1. Preprocesses the input data.
2. Loads the pre-trained models.
3. Outputs the prediction results to predictions.txt, and also generates a scatter plot, an evaluation metrics file metrics.json, and a comparison plot predicted_vs_actual.png. All output files are saved in the pre_output folder.

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/QingleCheng/RED-ACT-AI.git
   cd RED-ACT-AI
2. Prepare the environment:
    Install the required Python libraries mentioned in Prediction.py.
3. Place your input data in the root directory (e.g., example_input_data.csv).
4. Run the prediction script:
   python Prediction.py
5. The prediction results will be saved in predictions.txt.


### File Descriptions
- **`Prediction.py`**: Script to preprocess input data, load models, and predict damage indicators.
- **`example_input_data.csv`**: Example input data for testing the model.
- **`building_data.xlsx`**: Metadata of the 47 measured buildings.
- **`model/`**: Contains pre-trained models for predicting MIDR, MPFA, and PTFA.
### Acknowledgments
We appreciate the CESMD Strong Motion Database for providing the earthquake response data used in this project.

### Contact
If you have any questions or need further assistance, feel free to contact the author:
**Qingle Cheng**
Email: chengql94@163.com
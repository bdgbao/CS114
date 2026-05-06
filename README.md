# Parkinson Telemonitoring

## Project Structure

```text
CS114/
├── models/
│   ├── motor_updrs_model_RandomForest.pkl  # Model for predicting motor_UPDRS
│   ├── scaler.pkl                          # Scaler
│   └── total_updrs_model_RandomForest.pkl  # Model for predicting total_UPDRS
├── app.py                                  # Simple deployment using streamlit
├── complete_pipeline.ipynb                 # Complete pipeline for creating models, includes EDA, data preprocessing, model training and evaluation
├── preprocessed_data.csv                   # preprocessed data
├── README.md                               # yes, read him
└── requirements.txt                        # Required libraries
```

This is the repository of the final project for CS114, focused on Parkinson telemonitoring. The dataset can be found [here](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring). The complete pipeline is implemented in, obviously, `complete_pipeline.ipynb`. No extra download is required for running `complete_pipeline.ipynb`, you can just download the notebook, install the required libraries in `requirements.txt` and it's ready to go. For a simple local web demo, make sure you have installed all the requirements and run `streamlit run app.py` in your terminal.
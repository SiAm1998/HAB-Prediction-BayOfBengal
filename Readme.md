# HAB Prediction: Early Warning System for the Bay of Bengal

## Project Overview
This project develops a machine learning pipeline to forecast the risk of Harmful Algal Blooms (HABs) in the Northern Bay of Bengal. By integrating multi-parameter oceanographic data, the model provides an early warning system (5-7 days lead time) to support climate resilience and fisheries management.

## Key Features
- **Data Integration:** Merges Sea Surface Temperature (SST), Salinity, Nitrate, and Turbidity time-series data.
- **Machine Learning:** Utilizes Random Forest and XGBoost classifiers to predict HAB events.
- **Risk Assessment:** Classifies daily risk levels into Low, Medium, and High based on probabilistic outputs.

## Data Sources
The model uses historical time-series data derived from:
- **MODIS-Aqua Satellite Data:** For SST and Turbidity parameters.
- **In-situ Measurements:** For Nitrate and Salinity validation.
- **Target Variable:** Historical HAB event logs (Binary: 0=No Bloom, 1=Bloom).

## Methodology
1. **Preprocessing:** Temporal alignment of datasets and handling of missing values.
2. **Training:** Supervised classification using `RandomForestClassifier` with class balancing.
3. **Evaluation:** Validated using Stratified K-Fold Cross-Validation and Confusion Matrices.

## Results
The model achieves robust performance in distinguishing bloom events from non-bloom conditions, with `Nitrate` and `SST` identified as the top predictors of algal proliferation in this region.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
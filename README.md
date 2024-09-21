# ML Testing Project

This project demonstrates a simple machine learning testing setup using Python, Giskard, and Deepchecks.

## Setup

1. Install required packages:
    pip install -r requirements.txt


2. Run the model:
    python src/model.py

3. Run the tests:
    python src/test_model.py


## Project Structure

- `data/`: Contains the dataset (Iris dataset is used from scikit-learn)
- `src/`: Contains the source code
  - `model.py`: Defines and trains the model
  - `test_model.py`: Contains tests using Giskard and Deepchecks
- `requirements.txt`: Lists all required packages
- `README.md`: This file

## Testing

This project uses Giskard for performance and bias testing, and Deepchecks for model error analysis.


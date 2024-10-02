# ML Testing Project

[![CI](https://github.com/Artemon-line/ml_testing_project/actions/workflows/ci.yml/badge.svg)](https://github.com/Artemon-line/ml_testing_project/actions/workflows/ci.yml)

This project demonstrates a simple machine learning testing setup using Python, Giskard, and Deepchecks.

## Setup

1. Install required packages:
    pip install -r requirements.txt


2. Run the model:
    python src/model.py

3. Run the tests:
    python src/test_model.py


## CI/CD

This project uses GitHub Actions for continuous integration. The workflow is defined in `.github/workflows/ci.yml`. It runs automatically on pushes to the main branch and on pull requests.

The CI pipeline does the following:
1. Sets up a Python 3.9 environment
2. Installs project dependencies
3. Runs linting checks using flake8
4. Runs the tests

You can see the current status of the CI pipeline in the badge at the top of this README.

## Project Structure

- `data/`: Contains the dataset (Iris dataset is used from scikit-learn)
- `src/`: Contains the source code
  - `model.py`: Defines and trains the model
  - `test_model.py`: Contains tests using Giskard and Deepchecks
- `requirements.txt`: Lists all required packages
- `README.md`: This file

## Testing

This project uses Giskard for performance and bias testing, and Deepchecks for model error analysis.


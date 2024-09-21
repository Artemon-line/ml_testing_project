import pandas as pd
import numpy as np
from giskard import Model, Dataset
from deepchecks.tabular import Dataset as DeepcheckDataset
from deepchecks.tabular.checks import ModelErrorAnalysis
from model import train_model, load_data

def test_with_giskard():
    model, X_test, y_test = train_model()
    
    giskard_dataset = Dataset(X_test, name="iris_test")
    giskard_model = Model(
        model.predict,
        model.predict_proba,
        feature_names=X_test.columns,
        classification_labels=model.classes_
    )

    results = giskard_model.test_performance(giskard_dataset)
    print("Giskard Performance Test Results:")
    print(results)

    bias_test = giskard_model.test_bias(
        giskard_dataset,
        protected_feature="petal length (cm)",
        target="target"
    )
    print("\nGiskard Bias Test Results:")
    print(bias_test)

def test_with_deepchecks():
    _, X_test, y_test = load_data()
    model, _, _ = train_model()

    dc_dataset = DeepcheckDataset(X_test, label=y_test, cat_features=[])

    check = ModelErrorAnalysis()
    result = check.run(dc_dataset, model)
    
    print("\nDeepchecks Model Error Analysis:")
    result.show()

if __name__ == "__main__":
    test_with_giskard()
    test_with_deepchecks()

from giskard import Model, Dataset
from deepchecks.tabular import Dataset as DeepcheckDataset
from deepchecks.tabular.checks.model_evaluation import ConfusionMatrixReport
from model import train_model, load_data


def test_with_giskard():
    model, X_test, y_test = train_model()

    giskard_dataset = Dataset(X_test, name="iris_test")
    giskard_model = Model(
        model.predict,
        model.predict_proba,
        classification_labels=model.classes_,
        feature_names=list(X_test.columns),
        classification_threshold=0.5,
    )

    results = giskard_model.test_performance(giskard_dataset)
    print("Giskard Performance Test Results:")
    print(results)

    bias_test = giskard_model.test_bias(
        giskard_dataset, protected_feature="petal length (cm)", target="target"
    )
    print("\nGiskard Bias Test Results:")
    print(bias_test)


def test_with_deepchecks():
    _, X_test, _, y_test = load_data()
    model, _, _ = train_model()

    dc_dataset = DeepcheckDataset(X_test, label=y_test, cat_features=[])

    check = ConfusionMatrixReport()
    result = check.run(dc_dataset, model)

    print("\nDeepchecks Confusion Matrix Report:")
    result.show()


if __name__ == "__main__":
    test_with_giskard()
    test_with_deepchecks()

from eo_extension_gpa import ThresholdOptimizerIncludingLevellingDown

from fairlearn.postprocessing import ThresholdOptimizer
from error_parity import RelaxedThresholdOptimizer
from folktables import ACSDataSource, ACSEmployment
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from guardian_ai.fairness.bias_mitigation import EOModelBiasMitigator


# Loads ACSEmployment with Alaska only for debugging purposes
def get_dataset_numpy():
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    acs_data = data_source.get_data(download=True, states=["AL"])
    features, label, group = ACSEmployment.df_to_numpy(acs_data)

    # Filter to only keep 3 groups for simplicity
    group_filter = group <= 3
    features = features[group_filter]
    label = label[group_filter]
    group = group[group_filter]

    # 70-15-15 train-valid-test split
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        features, label, group, test_size=0.3, random_state=42
    )
    X_valid, X_test, y_valid, y_test, group_valid, group_test = train_test_split(
        X_test, y_test, group_test, test_size=0.5, random_state=42
    )

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        group_train,
        group_valid,
        group_test,
    )


def get_dataset_pandas():
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    acs_data = data_source.get_data(download=True, states=["AL"])
    features, label, group = ACSEmployment.df_to_pandas(acs_data)

    # Filter to only keep 3 groups for simplicity
    group_filter = (group <= 3).to_numpy()
    features = features[group_filter]
    label = label[group_filter]
    group = group[group_filter]

    # 70-15-15 train-valid-test split
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        features, label, group, test_size=0.3, random_state=42
    )
    X_valid, X_test, y_valid, y_test, group_valid, group_test = train_test_split(
        X_test, y_test, group_test, test_size=0.5, random_state=42
    )

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        group_train,
        group_valid,
        group_test,
    )


def train_base_model(X_train, y_train):
    model = RandomForestClassifier(random_state=10)
    model.fit(X_train, y_train)

    return model


def eo_postprocessing(base_model, X_valid, y_valid, group_valid, fairness_metric):
    postprocess_est = ThresholdOptimizer(
        estimator=base_model,
        constraints=fairness_metric,
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    postprocess_est.fit(X_valid, y_valid, sensitive_features=group_valid)

    return postprocess_est


def guardian_ai_postprocessing(
    base_model, X_valid, y_valid, group_valid, fairness_metric, accuracy_metric
):
    postprocess_est = EOModelBiasMitigator(
        base_estimator=base_model,
        protected_attribute_names=group_valid.columns[0],
        fairness_metric=fairness_metric,
        accuracy_metric=accuracy_metric,
    )

    postprocess_est.fit(X_valid, y_valid)

    return postprocess_est


def eo_gpa_postprocessing(base_model, X_valid, y_valid, group_valid, fairness_metric):
    postprocess_est = ThresholdOptimizerIncludingLevellingDown(
        estimator=base_model,
        constraints=fairness_metric,
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    postprocess_est.fit(X_valid, y_valid, sensitive_features=group_valid)

    return postprocess_est


def unproc_eo_postprocessing(
    base_model, X_valid, y_valid, group_valid, fairness_metric
):
    postproc_clf = RelaxedThresholdOptimizer(
        predictor=base_model,
        constraint=fairness_metric,
        tolerance=0.05,
    )

    # This code needs groups to start their ordering at 0
    y_scores = base_model.predict_proba(X_valid)[:, -1]
    postproc_clf.fit(X=X_valid, y=y_valid, group=group_valid - 1, y_scores=y_scores)

    return postproc_clf


def main_others():
    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        group_train,
        group_valid,
        group_test,
    ) = get_dataset_numpy()

    model = train_base_model(X_train, y_train)

    # eo_postprocessing(
    #     model, X_valid, y_valid, group_valid, fairness_metric="equalized_odds"
    # )
    # eo_gpa_postprocessing(
    #     model,
    #     X_valid,
    #     y_valid,
    #     group_valid,
    #     fairness_metric="false_negative_rate_parity",
    # )
    # unproc_eo_postprocessing(
    #     model, X_valid, y_valid, group_valid, fairness_metric="equalized_odds"
    # )


def main_guardian_ai():
    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        group_train,
        group_valid,
        group_test,
    ) = get_dataset_pandas()

    model = train_base_model(X_train, y_train)
    postproc_clf = guardian_ai_postprocessing(
        model,
        X_valid,
        y_valid,
        group_valid,
        fairness_metric="TPR",
        accuracy_metric="balanced_accuracy",
    )

    y_pred = postproc_clf.predict(X_test)
    print(f1_score(y_test, y_pred))


if __name__ == "__main__":
    # main_others() was used to better understand and compare
    # the two EO variants. It's not needed anymore now.
    # main_others()
    main_guardian_ai()

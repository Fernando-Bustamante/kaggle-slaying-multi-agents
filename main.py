import gc
import sys
import pandas as pd
import numpy as np
import yaml
from agents.config_agent import ConfigAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.modeling_agent import ModelingAgent
from agents.submission_agent import SubmissionAgent


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    print("=== Kaggle Multi-Agent System Starting ===\n")

    if len(sys.argv) > 1:
        competition_name = sys.argv[1]
    else:
        competition_name = input("Enter competition name (or press Enter to use config.yaml): ").strip()

    if competition_name:
        config_agent = ConfigAgent(competition_name)
        config = config_agent.build_config()

        # Orchestrator lê os dados e usa Claude para decidir a estratégia
        data_dir = config["output"]["data_dir"]

        # Count actual rows efficiently so orchestrator knows the true dataset size
        with open(f"{data_dir}train.csv", "rb") as _f:
            actual_n_rows = sum(1 for _ in _f) - 1  # subtract header
        print(f"[main] Actual training set size: {actual_n_rows:,} rows")

        probe = pd.read_csv(f"{data_dir}train.csv", nrows=5000)
        orchestrator = OrchestratorAgent(competition_name)
        decisions = orchestrator.decide(probe, config, actual_n_rows=actual_n_rows)
        del probe
        gc.collect()

        config["competition"]["task_type"] = decisions["task_type"]
        config["competition"]["metric"] = decisions["metric"]
        config["competition"]["is_timeseries"] = bool(decisions.get("is_timeseries", False))
        config["model"]["n_trials"] = int(decisions["n_trials"])
        config["model"]["cv_folds"] = int(decisions["cv_folds"])
        if decisions.get("sample_size") is not None:
            config["model"]["sample_size"] = int(decisions["sample_size"])
        if decisions.get("feature_selection_pct") is not None:
            config["model"]["feature_selection_pct"] = float(decisions["feature_selection_pct"])
        if decisions.get("n_ensemble_models") is not None:
            config["model"]["n_ensemble_models"] = int(decisions["n_ensemble_models"])
        if decisions.get("max_num_leaves") is not None:
            config["model"]["max_num_leaves"] = int(decisions["max_num_leaves"])
        if decisions.get("min_child_samples_min") is not None:
            config["model"]["min_child_samples_min"] = int(decisions["min_child_samples_min"])
        config["model"]["algorithm"] = decisions.get("algorithm", "lgbm")
        config["model"]["time_budget_minutes"] = int(decisions.get("time_budget_minutes", 90))
        print(f"[main] algorithm={config['model']['algorithm']} | time_budget={config['model']['time_budget_minutes']}min")

        # Hard safety overrides for large datasets — prevent runaway runtimes
        if actual_n_rows > 50_000:
            if config["model"].get("sample_size") is None:
                cap = min(50_000, actual_n_rows // 4)
                config["model"]["sample_size"] = cap
                print(f"[main] Large dataset safety: capping tuning sample to {cap:,} rows")
            if config["model"]["cv_folds"] > 5:
                config["model"]["cv_folds"] = 5
                print(f"[main] Large dataset safety: capping cv_folds to 5")
            if config["model"].get("algorithm") == "diverse":
                config["model"]["algorithm"] = "lgbm"
                print(f"[main] Large dataset safety: switching diverse → lgbm (diverse is for small datasets only)")
    else:
        config = load_config()

    # Step 1: Download and analyze data
    data_agent = DataAgent(config)
    if not competition_name:
        data_agent.download_data()
    train_sample, test = data_agent.load_data(sample=config["model"]["sample_size"])

    # Safety override: enforce diverse ensemble and conservative params for small datasets
    n_rows = len(train_sample)
    if n_rows < 2000:
        config["model"]["algorithm"] = "diverse"
        config["model"]["max_num_leaves"] = 20
        config["model"]["min_child_samples_min"] = max(30, n_rows // 25)
        if n_rows < 1000:
            # Very small: LR + RF only — LightGBM memorizes at this scale
            config["model"]["n_ensemble_models"] = 2
            print(f"[main] Very small dataset ({n_rows} rows) → diverse, LR+RF only")
        else:
            # Small: LR + RF + conservative LightGBM
            config["model"]["n_ensemble_models"] = 3
            print(f"[main] Small dataset ({n_rows} rows) → diverse, LR+RF+LightGBM (conservative)")
    data_agent.analyze(train_sample)

    # Auto-detect log transform: regression targets that are right-skewed and strictly positive
    # benefit from log1p — aligns local RMSE with Kaggle's RMSLE metric
    log_transform = False
    if config["competition"].get("task_type") == "regression":
        target_col = config["competition"]["target_column"]
        _y = train_sample[target_col].dropna()
        if _y.min() > 0 and float(_y.skew()) > 0.75:
            log_transform = True
            print(f"[main] Skewed target detected (skew={_y.skew():.2f}) → log1p transform enabled; "
                  f"OOF score will match Kaggle's RMSLE scale")
        del _y

    # Step 2: Limpa, cria features e encode no sample (para tuning)
    feature_agent = FeatureAgent(config)
    train_sample, test = feature_agent.add_delimiter_features(train_sample, test, fit=True)
    train_sample, test = feature_agent.drop_high_cardinality(train_sample, test)
    # impute_numeric=False: numeric NaNs are filled inside each CV fold to avoid leakage
    train_sample = feature_agent.clean(train_sample, fit=True, impute_numeric=False)
    test = feature_agent.clean(test, fit=False, impute_numeric=True)  # test gets full imputation for inference
    train_sample = feature_agent.add_statistical_features(train_sample)
    test = feature_agent.add_statistical_features(test)
    train_sample = feature_agent.log_transform_skewed_features(train_sample, fit=True)
    test = feature_agent.log_transform_skewed_features(test, fit=False)
    train_sample = feature_agent.add_datetime_features(train_sample, fit=True)
    test = feature_agent.add_datetime_features(test, fit=False)
    train_sample, test = feature_agent.add_target_encoding(train_sample, test)
    train_sample, test = feature_agent.encode(train_sample, test)
    train_sample, test = feature_agent.add_interaction_features(train_sample, test)

    # Feature selection via F-score (no model = no leakage, uses all data)
    train_sample, test = feature_agent.select_features(train_sample, test)
    tune_sample = train_sample
    del train_sample
    gc.collect()

    X_sample, y_sample = feature_agent.split_features_target(tune_sample)
    if log_transform:
        y_sample = np.log1p(y_sample)

    # Step 3: Tuning com sample
    modeling_agent = ModelingAgent(config)
    best_params = modeling_agent.tune(X_sample, y_sample)

    # Libera memória do sample antes do treino final
    del tune_sample, X_sample, y_sample
    gc.collect()

    # Step 4: Treino final com todos os dados
    print("\n[ModelingAgent] Loading full dataset for final training...")
    train_full, test = data_agent.load_data(sample=None)
    train_full, test = feature_agent.add_delimiter_features(train_full, test, fit=False)
    train_full, test = feature_agent.drop_high_cardinality(train_full, test)
    train_full = feature_agent.clean(train_full, fit=True, impute_numeric=False)
    test = feature_agent.clean(test, fit=False, impute_numeric=True)
    train_full = feature_agent.add_statistical_features(train_full)
    test = feature_agent.add_statistical_features(test)
    train_full = feature_agent.log_transform_skewed_features(train_full, fit=True)
    test = feature_agent.log_transform_skewed_features(test, fit=False)
    train_full = feature_agent.add_datetime_features(train_full, fit=True)
    test = feature_agent.add_datetime_features(test, fit=False)
    train_full, test = feature_agent.add_target_encoding(train_full, test)
    train_full, test = feature_agent.encode(train_full, test)
    train_full, test = feature_agent.add_interaction_features(train_full, test)
    train_full, test = feature_agent.apply_feature_selection(train_full, test)  # reusa seleção do sample
    X_full, y_full = feature_agent.split_features_target(train_full)
    del train_full
    gc.collect()
    if log_transform:
        y_full = np.log1p(y_full)

    id_col = config["competition"]["id_column"]
    test_ids = test[id_col].reset_index(drop=True)
    test_features = test.drop(columns=[id_col])
    del test
    gc.collect()

    avg_probas = modeling_agent.train_final(X_full, y_full, best_params, test_features)
    del X_full, y_full, test_features
    gc.collect()

    # Step 5: Gera predições e submete
    submission_agent = SubmissionAgent(config)
    task_type = config["competition"].get("task_type", "binary_classification")
    predict_type = config["competition"].get("predict_type", "proba")
    if task_type == "multiclass_classification":
        predictions = avg_probas.argmax(axis=1) if predict_type == "class" else avg_probas.max(axis=1)
    elif predict_type == "class":
        bool_format = config["competition"].get("bool_format", False)
        threshold = modeling_agent.optimal_threshold
        predictions = (avg_probas >= threshold)
        predictions = predictions if bool_format else predictions.astype(int)
    elif log_transform:
        predictions = np.expm1(avg_probas)
    else:
        predictions = avg_probas
    submission_agent.create_submission(test_ids, predictions)
    submission_agent.submit()

    scale = "log scale = RMSLE" if log_transform else "original scale"
    print(f"\n=== Done! OOF score: {modeling_agent.best_score:.4f} ({scale}) ===")
    print(f"[main] Run 'python leaderboard.py {competition_name}' in a few minutes to see your position.")


if __name__ == "__main__":
    main()

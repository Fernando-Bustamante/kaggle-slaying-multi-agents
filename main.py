import gc
import yaml
from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.modeling_agent import ModelingAgent
from agents.submission_agent import SubmissionAgent


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    print("=== Kaggle Multi-Agent System Starting ===\n")
    config = load_config()

    # Step 1: Download and analyze data
    data_agent = DataAgent(config)
    data_agent.download_data()
    train_sample, test = data_agent.load_data(sample=config["model"]["sample_size"])
    data_agent.analyze(train_sample)

    # Step 2: Limpa, cria features e encode no sample (para tuning)
    feature_agent = FeatureAgent(config)
    train_sample = feature_agent.clean(train_sample, fit=True)   # salva medianas do treino
    test = feature_agent.clean(test, fit=False)                   # usa medianas do treino
    train_sample = feature_agent.add_statistical_features(train_sample)
    test = feature_agent.add_statistical_features(test)
    train_sample, test = feature_agent.encode(train_sample, test)
    X_sample, y_sample = feature_agent.split_features_target(train_sample)

    # Step 3: Tuning com sample
    modeling_agent = ModelingAgent(config)
    best_params = modeling_agent.tune(X_sample, y_sample)

    # Libera memória do sample antes do treino final
    del train_sample, X_sample, y_sample
    gc.collect()

    # Step 4: Treino final com todos os dados
    print("\n[ModelingAgent] Loading full dataset for final training...")
    train_full, test = data_agent.load_data(sample=None)          # carrega TUDO
    train_full = feature_agent.clean(train_full, fit=True)        # refita medianas no full
    test = feature_agent.clean(test, fit=False)
    train_full = feature_agent.add_statistical_features(train_full)
    test = feature_agent.add_statistical_features(test)
    train_full, test = feature_agent.encode(train_full, test)     # refita encoder no full
    X_full, y_full = feature_agent.split_features_target(train_full)
    del train_full
    gc.collect()

    best_model = modeling_agent.train_final(X_full, y_full, best_params)

    # Step 5: Gera predições e submete
    submission_agent = SubmissionAgent(config)
    id_col = config["competition"]["id_column"]
    test_features = test.drop(columns=[id_col])
    predictions = best_model.predict_proba(test_features)[:, 1]
    submission_agent.create_submission(test, predictions)
    submission_agent.submit()
    submission_agent.check_leaderboard()

    print(f"\n=== Done! Best AUC: {modeling_agent.best_score:.4f} ===")


if __name__ == "__main__":
    main()

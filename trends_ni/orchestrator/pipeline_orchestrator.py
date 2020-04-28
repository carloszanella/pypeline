from typing import List


class PipelineOrchestrator:
    def __init__(
            self,
            ds_builder: DatasetBuilder,
            model_trainer: ModelTrainer,
            splitter: TrainValSplitter = TrainValSplitter(),
            seed: int = 42,
            val_split: float = 0.2
    ):
        self.ds_builder = ds_builder
        self.model_trainer = model_trainer
        self.splitter = splitter
        self.seed = seed
        self.train_ids = TRAIN_IDS
        self.val_split = val_split

    def run_pipeline(self):
        np.random.seed(self.seed)

        # Split data
        train_ix, val_ix = self.splitter.split(self.val_split)

        # Build datasets:
        X_train, y_train, X_val, y_val = self.build_datasets(train_ix, val_ix)

        # Train model
        model = self.model_trainer.train_model(X_train, y_train)

        # Evaluate on validation set
        scores, weighted_score = Score.evaluate_predictions(y_val, model.predict(X_val))
        self.get_validation_metrics(scores, weighted_score)

    def build_datasets(self, train_ix: List[float], val_ix: List[float]) -> Tuple[pd.DataFrame]:
        train_ds_id = f"{self.ds_builder.version}_{len(train_ix)}_{self.seed}"
        val_ds_id = f"{self.ds_builder.version}_{len(val_ix)}_{self.seed}"

        X_train, y_train = self.ds_builder.build_dataset(train_ix, train_ds_id)
        X_val, y_val = self.ds_builder.build_dataset(val_ix, val_ds_id)

        return X_train, y_train, X_val, y_val

    def get_validation_metrics(self, scores: List[float], weighted_score: float):
        print("\nValidation scores")
        print("###############")
        print("MAE: ", scores)
        print("Weighted Score: ", weighted_score)

        return scores, weighted_score
import TrainAndEvaluation
import DataLoader
import wandb
from Model import save_model, load_model
import HyperParameters


def init_w_and_b(hyper_params):
    epochs = hyper_params['epochs']
    learning_rate = hyper_params['learning_rate']

    if HyperParameters.WB:
        wandb.init(
            # Set the project where this run will be logged
            group=f"{hyper_params['model_name']}",
            project="ASR",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"{hyper_params}",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "architecture": f"{hyper_params['model_name']}",
                "dataset": "AN4",
                "epochs": epochs,

            })


def run_model(hparams):
    if HyperParameters.WB:
        wandb.login()
        init_w_and_b(hparams)

    train_batch_iterator = DataLoader.get_batch_iterator("train",
                                                         hparams["batch_size"],
                                                         deletion_augmentations=hparams["deletion_augmentations"],
                                                         stretch_augmentation=hparams["stretch_augmentation"],
                                                         feat_type=hparams["feat_type"],
                                                         n_feats=hparams["n_feats"])
    test_batch_iterator = DataLoader.get_batch_iterator("test", hparams["batch_size"],
                                                        feat_type=hparams["feat_type"],
                                                        n_feats=hparams["n_feats"])
    val_batch_iterator = DataLoader.get_batch_iterator("val", hparams["batch_size"],
                                                       feat_type=hparams["feat_type"],
                                                       n_feats=hparams["n_feats"])
    all_iterators = [train_batch_iterator, test_batch_iterator, val_batch_iterator]
    return TrainAndEvaluation.train_and_validation(hparams, all_iterators)


def test_model(model, hparams,):
    if HyperParameters.WB:
        wandb.login()
        init_w_and_b(hparams)

    test_batch_iterator = DataLoader.get_batch_iterator("test", hparams["batch_size"],
                                                        feat_type=hparams["feat_type"],
                                                        n_feats=hparams["n_feats"])

    TrainAndEvaluation.test(hparams, test_batch_iterator, model)


if __name__ == '__main__':
    hparams = HyperParameters.deep_speech_hparams


    try:
        model = load_model(f"{hparams['model_name']}_model", f"{hparams['model_name']}_hparams")
        print("loaded model")
    except:
        print("could not load model")
        model = run_model(hparams)
        save_model(model, f"{hparams['model_name']}_model", HyperParameters.deep_speech_hparams)
    test_model(model, HyperParameters.deep_speech_hparams)



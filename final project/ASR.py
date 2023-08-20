import TrainAndEvaluation
import DataLoader
import wandb

import HyperParameters

DESCRIPTION = 'initial work'
RUN = 'Complex Model'


def init_w_and_b(hyper_params):
    epochs = hyper_params['epochs']
    learning_rate = hyper_params['learning_rate']

    if HyperParameters.WB:
        wandb.init(
            # Set the project where this run will be logged
            group="Complex Model initial work",
            project="ASR",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"{DESCRIPTION}{RUN}_{hyper_params}",
            notes='checking if log is work properly',
            # Track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "architecture": "assembly",
                "dataset": "AN4",
                "epochs": epochs,

            })


def run_deep_speech():
    if HyperParameters.WB:
        wandb.login()
        init_w_and_b(HyperParameters.deep_speech_hparams)

    train_batch_iterator = DataLoader.get_batch_iterator("train", HyperParameters.deep_speech_hparams["batch_size"])
    test_batch_iterator = DataLoader.get_batch_iterator("test", HyperParameters.deep_speech_hparams["batch_size"])
    val_batch_iterator = DataLoader.get_batch_iterator("val",  HyperParameters.deep_speech_hparams["batch_size"])
    all_iterators = [train_batch_iterator, test_batch_iterator, val_batch_iterator]
    TrainAndEvaluation.deep_speech_train_and_validation( HyperParameters.deep_speech_hparams, all_iterators)

    if  HyperParameters.WB:
        wandb.finish()


def run_cnn():
    if  HyperParameters.WB:
        wandb.login()
        init_w_and_b( HyperParameters.res_cnn_hparams)

    train_batch_iterator = DataLoader.get_batch_iterator("train", HyperParameters.deep_speech_hparams["batch_size"])
    test_batch_iterator = DataLoader.get_batch_iterator("test", HyperParameters.deep_speech_hparams["batch_size"])
    val_batch_iterator = DataLoader.get_batch_iterator("val", HyperParameters.deep_speech_hparams["batch_size"])
    all_iterators = [train_batch_iterator, test_batch_iterator, val_batch_iterator]
    TrainAndEvaluation.res_cnn_train_and_validation(HyperParameters.deep_speech_hparams, all_iterators)

    if HyperParameters.WB:
        wandb.finish()


if __name__ == '__main__':
    # run_deep_speech()
    run_cnn()

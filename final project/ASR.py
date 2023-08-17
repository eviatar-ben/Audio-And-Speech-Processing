import TrainAndEvaluation
import DataLoader
import wandb
from HyperParameters import hparams
from HyperParameters import WB

DESCRIPTION = 'initial work'
RUN = 'Complex Model'


def init_w_and_b():
    epochs = hparams['epochs']
    learning_rate = hparams['learning_rate']

    if WB:
        wandb.init(
            # Set the project where this run will be logged
            group="Complex Model initial work",
            project="ASR",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"{DESCRIPTION}{RUN}_{hparams}_epochs",
            notes='checking if log is work properly',
            # Track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "architecture": "assembly",
                "dataset": "AN4",
                "epochs": epochs,

            })


def main():
    if WB:
        wandb.login()
        init_w_and_b()

    train_batch_iterator = DataLoader.get_batch_iterator("train", hparams["batch_size"])
    test_batch_iterator = DataLoader.get_batch_iterator("test", hparams["batch_size"])
    val_batch_iterator = DataLoader.get_batch_iterator("val", hparams["batch_size"])
    all_iterators = [train_batch_iterator, test_batch_iterator, val_batch_iterator]

    TrainAndEvaluation.train_and_validation(hparams, all_iterators)

    if WB:
        wandb.finish()


if __name__ == '__main__':
    main()

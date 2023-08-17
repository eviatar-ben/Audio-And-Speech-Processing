import Model
import TrainAndEvaluation
import torchaudio
import load_data
import wandb

WB = True
DESCRIPTION = 'initial work'
RUN = 'Complex Model'

learning_rate = 5e-4
batch_size = 10
epochs = 10

hparams = {
    "n_cnn_layers": 3,
    "n_rnn_layers": 2,
    "rnn_dim": 128,
    "n_class": 29,
    "n_feats": 13,
    "stride": 2,
    "dropout": 0.1,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs
}


def init_w_and_b():
    if WB:
        wandb.init(
            # Set the project where this run will be logged
            group="Complex Model initial work",
            project="ASR",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"{DESCRIPTION}{RUN}_{epochs}_epochs",
            notes='checking if log is work properly',
            # Track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "architecture": "CNN",
                "dataset": "CIFAR-10",
                "epochs": epochs,

            })


def main():
    if WB:
        wandb.login()
        init_w_and_b()

    train_batch_iterator = load_data.get_batch_iterator("train", batch_size)
    test_batch_iterator = load_data.get_batch_iterator("test", batch_size)
    val_batch_iterator = load_data.get_batch_iterator("val", batch_size)
    all_iterators = [train_batch_iterator, test_batch_iterator, val_batch_iterator]

    TrainAndEvaluation.train_test_valid(hparams, all_iterators)

    if WB:
        wandb.finish()


if __name__ == '__main__':
    main()

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


def run_model(hparams):
    if HyperParameters.WB:
        wandb.login()
        init_w_and_b(hparams)

    train_batch_iterator, test_batch_iterator, val_batch_iterator = DataLoader.get_batch_iterator(hparams["batch_size"])
    all_iterators = [train_batch_iterator, test_batch_iterator, val_batch_iterator]
    TrainAndEvaluation.train_and_validation(hparams, all_iterators)
fi

if __name__ == '__main__':
    # run_model(HyperParameters.res_cnn_hparams)
    # run_model(HyperParameters.transformer_hparams)
    # run_model(HyperParameters.rnn_hparams)
    run_model(HyperParameters.deep_speech_hparams)
    # run_model(HyperParameters.listen_attend_spell_hparams)
    # run_model(HyperParameters.rnnt_hparams)
    # run_model(HyperParameters.multiTransformer_hparams)

WB = False

learning_rate = 5e-4
batch_size = 10
epochs = 200
delta = True
delta_delta = False

res_cnn_hparams = {
    "n_cnn_layers": 3,
    "n_class": 29,
    "n_feats": 13,
    "stride": 2,

    "dropout": 0.1,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs,
    "augmentations": True,
    "model_name": "res_cnn"
}

transformer_hparams = {
    "n_cnn_layers": 3,
    "n_class": 29,
    "n_feats": 13,
    "dropout": 0.1,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs,
    "augmentations": True,
    "model_name": "transformer"
}

rnn_hparams = {
    "n_cnn_layers": 3,
    "n_class": 29,
    "n_feats": 13,
    "dropout": 0.1,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs,
    "augmentations": True,
    "model_name": "rnn"
}

deep_speech_hparams = {
    "n_cnn_layers": 4,
    "n_rnn_layers": 3,
    "rnn_dim": 128,
    "n_class": 29,
    "feat_type": "mel_spectrogram",
    "n_feats": 40,
    "stride": 2,
    "dropout": 0.1,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs,
    "augmentations": True,
    "model_name": "deep_speech"
}

multiTransformer_hparams = {
    "n_cnn_layers": 3,
    "n_class": 29,
    "n_feats": 13,
    "dropout": 0.1,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs,
    "augmentations": True,
    "model_name": "multiTransformer"

}

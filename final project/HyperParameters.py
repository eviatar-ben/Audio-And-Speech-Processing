WB = False

learning_rate = 5e-4
batch_size = 10
epochs = 200

deep_speech_hparams = {
    "n_cnn_layers": 3,
    "n_rnn_layers": 2,
    "rnn_dim": 128,
    "n_class": 29,
    "n_feats": 13,
    "delta": False,
    "delta_delta": False,
    "stride": 2,

    "dropout": 0.1,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs,
    "model_name": "deep_speech"
}

res_cnn_hparams = {
    "n_cnn_layers": 12,
    "n_class": 29,
    "n_feats": 13,
    "stride": 2,
    "dropout": 0.1,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs,
    "model_name": "res_cnn"
}

config = {
    "model": {
        "batch_size": 32, #batch size of the model
        "embed_dim": 256, #embed dimension as used in the transformer model
        "ff_dim": 3, # feed forward dimension as used in the transformer model
        "train_cap": 600000, #the number of training samples to consider from train.csv
        "dev_cap": 400000, #the number of training samples to consider from dev.csv
        "epochs": 50 #number of epochs
    },

    "audio_data_path": {
        "train": "./LA/ASVspoof2019_LA_train/flac/",
        "test": "./LA/ASVspoof2019_LA_eval/flac/",
        "dev": "./LA/ASVspoof2019_LA_dev/flac/"
    },

    "data_path": {
        "train": "./Dataset/train.csv",
        "test": "./Dataset/test.csv",
        "dev": "./Dataset/dev.csv"
    },

    "spectogram": {
        "sr": None,
        "n_fft": 512,
        "hop_length": 128,
        "n_mels": 165,
        "fmin": 20,
        "fmax": 8300,
        "top_db": 80  
    },

    "model_path": {
        "embed": "./Model/Embed/",
        "transformer": "./Model/Transformer/"
    },

    "results": {
        "path": "./Results/"
    }

}
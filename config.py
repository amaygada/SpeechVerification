config = {
    "model": {
        "batch_size": 8,
        "embed_dim": 256,
        "ff_dim": 3,
        "train_cap": 16,
        "dev_cap": 16,
        "epochs": 3
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
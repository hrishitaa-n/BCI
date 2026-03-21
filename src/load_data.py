import mne


def load_data(file_path):
    print("Loading EEG data...")
    raw = mne.io.read_raw_gdf(file_path, preload=True)
    return raw

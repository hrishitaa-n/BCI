import mne

from src.load_data import load_data
from src.preprocess import bandpass_filter
from src.model import train_svm

# File path
file_path = "data/BCICIV_2a_gdf/A01T.gdf"

print("=== BCI Pipeline Started ===")

# Load data
raw = load_data(file_path)

# Filter
raw = bandpass_filter(raw)

# Events
events, event_dict = mne.events_from_annotations(raw)

event_id = {
    '769': event_dict['769'],
    '770': event_dict['770']
}

# Epochs
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=1,
    tmax=4,
    baseline=None,
    preload=True
)

# Data
X = epochs.get_data()
y = epochs.events[:, -1]
y = y - min(y)

# Model (CSP is inside now)
accuracy = train_svm(X, y)

print("Final Accuracy:", accuracy)
print("=== Pipeline Complete ===")

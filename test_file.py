from src.model import train_svm
import mne
from src.preprocess import bandpass_filter
import os
import sys
from src.load_data import load_data
from src.features import apply_csp

file_path = "data/BCICIV_2a_gdf/A01T.gdf"

raw = load_data(file_path)

print(raw)
sys.path.append(os.path.abspath("."))


file_path = "data/BCICIV_2a_gdf/A01T.gdf"

# Load
raw = load_data(file_path)

# Filter
raw = bandpass_filter(raw)

print(raw)


raw = load_data(file_path)
raw = bandpass_filter(raw)

# Extract events
events, _ = mne.events_from_annotations(raw)

print(events[:10])

events, event_dict = mne.events_from_annotations(raw)

event_id = {
    '769': event_dict['769'],  # left
    '770': event_dict['770']   # right
}

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=1,
    tmax=4,
    baseline=None,
    preload=True
)

print(epochs)


# Get data
X = epochs.get_data()
y = epochs.events[:, -1]

# Convert labels to 0 and 1
y = y - min(y)

print("Before CSP shape:", X.shape)

# Apply CSP
X_csp, csp = apply_csp(X, y)

print("After CSP shape:", X_csp.shape)


accuracy = train_svm(X_csp, y)
print("Final Accuracy:", accuracy)

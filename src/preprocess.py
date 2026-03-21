def bandpass_filter(raw, l_freq=8, h_freq=30):
    print("Applying bandpass filter (8–30 Hz)...")
    raw.filter(l_freq, h_freq)
    return raw

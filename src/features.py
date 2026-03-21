from mne.decoding import CSP


def apply_csp(X, y, n_components=4):
    print("Applying CSP...")

    csp = CSP(n_components=n_components)
    X_csp = csp.fit_transform(X, y)

    return X_csp, csp

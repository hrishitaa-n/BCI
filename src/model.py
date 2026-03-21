from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from mne.decoding import CSP


def train_svm(X, y):
    model = Pipeline([
        ('csp', CSP(n_components=4)),
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf'))
    ])

    scores = cross_val_score(model, X, y, cv=5)

    return scores.mean()

"""Train both Koopman backbones and save pickles (for the article runs)."""
import os, pickle, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import train as T

here = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(here, 'data', 'fhn_koopman_t80.npz'))

CONFIGS = [
    dict(backbone='spiral',   m=6, epochs=1000, batch=32, lr=1e-3),
    dict(backbone='bilinear', m=4, epochs=500,  batch=24, lr=1e-3),
]

for c in CONFIGS:
    t0 = time.time()
    print(f"\n==== training {c} ====", flush=True)
    out = T.train(data, verbose=True, **c)
    path = os.path.join(here, 'data', f"koopman_{c['backbone']}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(out, f)
    print(f"==== saved {path}  ({time.time()-t0:.0f}s) ====", flush=True)

print("\nALL TRAINING DONE", flush=True)

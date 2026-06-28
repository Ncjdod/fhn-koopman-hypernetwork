"""SINDYc -- Sparse Identification of Nonlinear Dynamics with control."""
import numpy as np

def _poly_feature_names(degree):
    """Names for the autonomous polynomial part in (v, w) up to `degree`."""
    names = ["1"]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            j = d - i
            term = ("v" * i) + ("w" * j)
            names.append(term)
    return names

def feature_names(degree=3, control_cross=True):
    """Full library column names: poly(v,w) then control columns."""
    names = _poly_feature_names(degree)
    ctrl = ["u"]
    if control_cross:
        ctrl += ["u*v", "u*w"]
    return names + ctrl

def build_library(V, W, U, degree=3, control_cross=True):
    """Library Theta (N, n_features) for states V,W and control U (each (N,))."""
    V = np.asarray(V); W = np.asarray(W); U = np.asarray(U)
    cols = [np.ones_like(V)]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            j = d - i
            cols.append((V ** i) * (W ** j))
    cols.append(U)
    if control_cross:
        cols.append(U * V)
        cols.append(U * W)
    return np.stack(cols, axis=1)

def finite_difference(Y, dt):
    """4th-order-accurate central difference of a trajectory Y (T, d) -> (T, d)."""
    Y = np.asarray(Y, dtype=float)
    d = np.empty_like(Y)
    d[2:-2] = (-Y[4:] + 8 * Y[3:-1] - 8 * Y[1:-3] + Y[:-4]) / (12 * dt)
    d[1] = (Y[2] - Y[0]) / (2 * dt)
    d[-2] = (Y[-1] - Y[-3]) / (2 * dt)
    d[0] = (-3 * Y[0] + 4 * Y[1] - Y[2]) / (2 * dt)
    d[-1] = (3 * Y[-1] - 4 * Y[-2] + Y[-3]) / (2 * dt)
    return d

def _ridge(A, b, lam):
    """Ridge least squares argmin ||A x - b||^2 + lam ||x||^2."""
    n = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ b)

def stlsq(Theta, dXdt, threshold=0.05, n_iter=20, ridge=1e-6):
    """Sequentially-thresholded least squares, one column of Xi per output."""
    Theta = np.asarray(Theta, dtype=float)
    dXdt = np.asarray(dXdt, dtype=float)
    norms = np.linalg.norm(Theta, axis=0)
    norms = np.where(norms < 1e-12, 1.0, norms)
    Tn = Theta / norms

    n_feat = Tn.shape[1]
    n_out = dXdt.shape[1]
    Xi = np.zeros((n_feat, n_out))
    for k in range(n_out):
        y = dXdt[:, k]
        w = _ridge(Tn, y, ridge)
        active = np.ones(n_feat, dtype=bool)
        for _ in range(n_iter):
            new_active = np.abs(w) >= threshold
            if _ > 0 and np.array_equal(new_active, active):
                break
            active = new_active
            w = np.zeros(n_feat)
            if active.any():
                w[active] = _ridge(Tn[:, active], y, ridge)
        Xi[:, k] = w
    return Xi / norms[:, None]

class SINDYcModel:
    """Identified control-affine polynomial ODE ẋ = Theta(x) · Xi."""

    def __init__(self, Xi, degree=3, control_cross=True, dt=0.05):
        self.Xi = np.asarray(Xi)
        self.degree = degree
        self.control_cross = control_cross
        self.dt = dt
        self.names = feature_names(degree, control_cross)
        self.ctrl_idx = [i for i, n in enumerate(self.names) if "u" in n]
        self.auto_idx = [i for i, n in enumerate(self.names) if "u" not in n]

    def _auto_lib(self, V, W):
        return build_library(V, W, np.zeros_like(V), self.degree, self.control_cross)

    def f(self, x):
        """Autonomous part f(x) (control set to 0), x (...,2) -> (...,2)."""
        x = np.atleast_2d(x)
        Th = self._auto_lib(x[:, 0], x[:, 1])
        Th[:, self.ctrl_idx] = 0.0
        return Th @ self.Xi

    def gain(self, x):
        """d(ẋ)/du at control 0: the affine control gain g(x), (...,2)."""
        x = np.atleast_2d(x)
        V, W = x[:, 0], x[:, 1]
        g = np.zeros((x.shape[0], len(self.names)))
        for i, n in enumerate(self.names):
            if n == "u":
                g[:, i] = 1.0
            elif n == "u*v":
                g[:, i] = V
            elif n == "u*w":
                g[:, i] = W
        return g @ self.Xi

    def vector_field(self, x, u):
        """ẋ = f(x) + g(x)·u."""
        x = np.atleast_2d(x)
        u = np.broadcast_to(np.asarray(u, dtype=float), (x.shape[0],))
        return self.f(x) + self.gain(x) * u[:, None]

    def simulate(self, x0, u_seq, dt=None):
        """RK4 rollout."""
        dt = self.dt if dt is None else dt
        x0 = np.atleast_2d(np.asarray(x0, dtype=float))
        B = x0.shape[0]
        u_seq = np.asarray(u_seq, dtype=float)
        if u_seq.ndim == 1:
            u_seq = np.broadcast_to(u_seq, (B, u_seq.shape[0]))
        T = u_seq.shape[1]
        out = np.empty((B, T + 1, 2))
        out[:, 0] = x0
        x = x0
        for t in range(T):
            u = u_seq[:, t]
            k1 = self.vector_field(x, u)
            k2 = self.vector_field(x + 0.5 * dt * k1, u)
            k3 = self.vector_field(x + 0.5 * dt * k2, u)
            k4 = self.vector_field(x + dt * k3, u)
            x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            out[:, t + 1] = x
        return out

    def invert_onestep(self, x, xdot_desired, u_lo=-5.0, u_hi=5.0, eps=1e-6):
        """Current that yields a desired instantaneous v̇ (the v-equation carries the control in FHN)."""
        x = np.atleast_2d(x)
        fv = self.f(x)[:, 0]
        gv = self.gain(x)[:, 0]
        u = (np.asarray(xdot_desired) - fv) / (gv + np.sign(gv + eps) * eps)
        return np.clip(u, u_lo, u_hi)

    def equations(self, thr=1e-4):
        out = []
        for k, lab in enumerate(["v_dot", "w_dot"]):
            terms = [f"{self.Xi[i,k]:+.4f}·{self.names[i]}"
                     for i in range(len(self.names)) if abs(self.Xi[i, k]) > thr]
            out.append(f"{lab} = " + " ".join(terms))
        return out

    def cubic_coeff(self):
        """Coefficient of v³ in the v̇ equation (must be < 0 for boundedness)."""
        i = self.names.index("vvv") if "vvv" in self.names else None
        return float(self.Xi[i, 0]) if i is not None else None

    def assert_bounded(self, u_test=(0.0, 0.5, 1.0), periods=200, dt=None,
                       period_steps=740):
        """Long free rollout at several constant currents -> must stay finite and in a bounded annulus (neither |x|->inf nor a spurious blow-up)."""
        dt = self.dt if dt is None else dt
        report = {}
        c3 = self.cubic_coeff()
        for u in u_test:
            roll = self.simulate(np.array([-1.0, -0.5]),
                                 np.full(periods * period_steps, u), dt)
            v = roll[0, :, 0]
            report[u] = {"finite": bool(np.isfinite(v).all()),
                         "max_abs": float(np.max(np.abs(v))),
                         "tail_ptp": float(v[len(v)//2:].max() - v[len(v)//2:].min())}
        return {"v3_coeff": c3, "v3_negative": (c3 is not None and c3 < 0),
                "rollouts": report}

def fit(ys, us, dt, degree=3, control_cross=True, threshold=0.05,
        ridge=1e-6, use_exact_derivatives=False):
    """Fit a SINDYc model."""
    ys = np.asarray(ys, dtype=float)
    us = np.asarray(us, dtype=float)
    B, T, _ = ys.shape

    if use_exact_derivatives:
        from data_gen import fhn_derivatives
        import jax.numpy as jnp
        dots = np.asarray(fhn_derivatives(jnp.asarray(ys), jnp.asarray(us)))
        V = ys[..., 0].reshape(-1); W = ys[..., 1].reshape(-1)
        U = us.reshape(-1); dX = dots.reshape(-1, 2)
    else:
        dV = np.concatenate([finite_difference(ys[b], dt) for b in range(B)], axis=0)
        V = ys[..., 0].reshape(-1); W = ys[..., 1].reshape(-1)
        U = us.reshape(-1); dX = dV

    Theta = build_library(V, W, U, degree, control_cross)
    Xi = stlsq(Theta, dX, threshold=threshold, ridge=ridge)
    return SINDYcModel(Xi, degree=degree, control_cross=control_cross, dt=dt)

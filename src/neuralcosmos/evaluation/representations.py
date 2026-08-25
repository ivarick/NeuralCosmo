"""Frozen-representation extraction and probing.

Plan reference: sections 36, 37, 38, 39.

Two questions are asked of a trained encoder, and section 37 insists they be
answered together:

  domain probe  -- how much simulator identity survives in the representation?
  target probe  -- how much cosmological information survives?

Reporting either alone is misleading. A collapsed representation hides the
simulator perfectly while being useless, so a near-chance domain probe is only
good news if the target probe is still strong.

LEAKAGE
-------
Probes are fitted and scored on DISJOINT SIMULATIONS, exactly as the main task
is. Fifteen maps of one simulation are near-duplicates; splitting them randomly
would let a probe memorise a simulation and be scored on its own training data,
inflating both scores and making the whole diagnostic meaningless.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "extract_embeddings",
    "simulation_level_split",
    "fit_domain_probe",
    "fit_target_probe",
]


def extract_embeddings(model, dataset, device, batch_size: int = 64) -> dict[str, np.ndarray]:
    """Run the frozen encoder over a dataset and collect latent vectors."""
    import torch
    from torch.utils.data import DataLoader

    model = model.to(device).eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=dataset.safe_num_workers(0), pin_memory=device.type == "cuda",
    )

    z, suites, sims, maps, targets = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            feats = model.backbone(x)
            z.append(feats.float().cpu().numpy())
            suites.append(batch["suite_id"].numpy())
            sims.append(batch["simulation_id"].numpy())
            maps.append(batch["map_id"].numpy())
            targets.append(batch["target"].numpy())

    return {
        "z": np.concatenate(z),
        "suite_id": np.concatenate(suites),
        "simulation_id": np.concatenate(sims),
        "map_id": np.concatenate(maps),
        "target": np.concatenate(targets),
    }


def simulation_level_split(
    suite_ids: np.ndarray,
    simulation_ids: np.ndarray,
    test_fraction: float = 0.4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks splitting samples by (suite, simulation), never by map.

    The key is composed from suite and simulation because simulation 7 of one
    suite is unrelated to simulation 7 of another.
    """
    rng = np.random.default_rng(seed)
    keys = np.stack([suite_ids, simulation_ids], axis=1)
    unique = np.unique(keys, axis=0)

    n_test = max(1, int(round(len(unique) * test_fraction)))
    order = rng.permutation(len(unique))
    test_keys = {tuple(unique[i]) for i in order[:n_test]}

    is_test = np.array([tuple(k) in test_keys for k in keys], dtype=bool)
    return ~is_test, is_test


def fit_domain_probe(
    z: np.ndarray,
    suite_ids: np.ndarray,
    simulation_ids: np.ndarray,
    seed: int = 0,
    max_iter: int = 2000,
) -> dict[str, Any]:
    """Predict simulator identity from a frozen representation (section 37)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    train, test = simulation_level_split(suite_ids, simulation_ids, seed=seed)
    n_classes = len(np.unique(suite_ids))
    chance = 1.0 / n_classes

    scaler = StandardScaler().fit(z[train])
    z_tr, z_te = scaler.transform(z[train]), scaler.transform(z[test])
    y_tr, y_te = suite_ids[train], suite_ids[test]

    out: dict[str, Any] = {
        "n_classes": n_classes,
        "chance_accuracy": chance,
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "probes": {},
    }

    probes = {
        "linear": LogisticRegression(max_iter=max_iter, random_state=seed),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128,), max_iter=3000, random_state=seed,
            early_stopping=True, n_iter_no_change=40, tol=1e-5,
        ),
    }
    for name, clf in probes.items():
        clf.fit(z_tr, y_tr)
        pred = clf.predict(z_te)
        entry = {
            "accuracy": float((pred == y_te).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        }
        if n_classes == 2:
            proba = clf.predict_proba(z_te)[:, 1]
            entry["auroc"] = float(roc_auc_score(y_te, proba))
        # How far above chance, as a fraction of the available headroom.
        entry["above_chance"] = float(
            (entry["balanced_accuracy"] - chance) / (1.0 - chance)
        )
        out["probes"][name] = entry

    return out


def fit_target_probe(
    z: np.ndarray,
    targets: np.ndarray,
    suite_ids: np.ndarray,
    simulation_ids: np.ndarray,
    target_names: list[str],
    seed: int = 0,
) -> dict[str, Any]:
    """Recover the cosmological targets from a frozen representation (section 38).

    Reported alongside the domain probe, never instead of it: a representation
    that hides the simulator by collapsing would score near chance on the domain
    probe and near zero here, and only the pair distinguishes that failure from
    success.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    train, test = simulation_level_split(suite_ids, simulation_ids, seed=seed)
    scaler = StandardScaler().fit(z[train])
    z_tr, z_te = scaler.transform(z[train]), scaler.transform(z[test])
    y_tr, y_te = targets[train], targets[test]

    out: dict[str, Any] = {"n_train": int(train.sum()), "n_test": int(test.sum()), "probes": {}}

    probes = {
        "linear": Ridge(alpha=1.0, random_state=seed),
        # A nonlinear probe must be given enough optimisation budget to at
        # least match the linear one. If it does not, it is reporting its own
        # underfitting rather than a property of the representation.
        "mlp": MLPRegressor(
            hidden_layer_sizes=(128,), max_iter=3000, random_state=seed,
            early_stopping=True, n_iter_no_change=40, tol=1e-5,
        ),
    }
    for name, reg in probes.items():
        reg.fit(z_tr, y_tr)
        pred = reg.predict(z_te)
        if pred.ndim == 1:
            pred = pred[:, None]
        out["probes"][name] = {
            n: {
                "r2": float(r2_score(y_te[:, i], pred[:, i])),
                "rmse": float(np.sqrt(np.mean((y_te[:, i] - pred[:, i]) ** 2))),
            }
            for i, n in enumerate(target_names)
        }
    return out

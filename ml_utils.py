import numpy as np
from functools import lru_cache
from scipy.special import gamma

@lru_cache(None)
def _precompute_gamma(alpha, terms):
    """
    Calcula y cachea los valores de Gamma(alpha*k + 1) para k=0..terms-1.
    """
    k = np.arange(terms)
    return gamma(alpha * k + 1)

def mittag_leffler(alpha, z, terms=50, threshold=1e-6):
    """
    Función de Mittag-Leffler E_{alpha,1}(z) ~ sum_{k=0..terms-1} z^k / Gamma(alpha*k + 1).
    """
    gamma_terms = _precompute_gamma(alpha, terms)
    k = np.arange(terms)
    z = np.atleast_1d(z)
    z_k = np.power.outer(z, k)
    terms_series = z_k / gamma_terms

    # (Opcional) Filtrado de términos muy pequeños:
    # terms_series[np.abs(terms_series) < threshold] = 0

    ml = np.sum(terms_series, axis=1)
    return ml

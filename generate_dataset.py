"""
Synthetic Galaxy Image Generator for Case Study
================================================
Generates realistic galaxy-like images for three morphological classes:
    - Elliptical (smooth, concentrated, ellipsoidal light profile)
    - Spiral    (disk + spiral arms with background bulge)
    - Irregular (clumpy, asymmetric brightness distribution)

This stand-in for SDSS / Galaxy Zoo images allows the case study to be
fully reproducible without large external downloads (~1.4 GB for Galaxy10).
The morphology generation models are physically motivated (Sersic profile
for ellipticals, exponential disk + logarithmic spiral arms for spirals,
and a Gaussian Mixture for irregulars).

For a production paper, one would replace this with:
    from astroNN.datasets import load_galaxy10
"""
import numpy as np
from pathlib import Path

IMG_SIZE = 64
RNG_DEFAULT_SEED = 42


def _sersic_profile(size, n=4.0, re=8.0, axis_ratio=0.7, pa_deg=30.0,
                    cx=None, cy=None):
    """Sersic light profile - models elliptical galaxies."""
    if cx is None:
        cx = size / 2
    if cy is None:
        cy = size / 2
    y, x = np.indices((size, size))
    pa = np.deg2rad(pa_deg)
    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    # Rotate coordinate frame
    xr = (x - cx) * cos_pa + (y - cy) * sin_pa
    yr = -(x - cx) * sin_pa + (y - cy) * cos_pa
    # Elliptical radius
    r = np.sqrt(xr ** 2 + (yr / axis_ratio) ** 2)
    # Sersic intensity
    bn = 2 * n - 1.0 / 3.0  # approximation
    intensity = np.exp(-bn * ((r / re) ** (1.0 / n) - 1.0))
    return intensity


def _spiral_arms(size, n_arms=2, pitch_deg=20.0, pa_deg=0.0,
                 disk_scale=10.0, bulge_strength=0.4):
    """Exponential disk + logarithmic spiral arms + central bulge."""
    cx, cy = size / 2, size / 2
    y, x = np.indices((size, size))
    dx, dy = x - cx, y - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    # Exponential disk
    disk = np.exp(-r / disk_scale)

    # Spiral arm enhancement (logarithmic spiral)
    pitch = np.deg2rad(pitch_deg)
    pa = np.deg2rad(pa_deg)
    arm_phase = n_arms * (theta - pa) - np.log(r + 1e-3) / np.tan(pitch)
    arms = (np.cos(arm_phase) + 1) / 2  # 0..1
    # Suppress arms in the very center
    arms = arms * (1 - np.exp(-r / 4.0))

    # Central bulge (small Sersic)
    bulge = _sersic_profile(size, n=2.0, re=3.0, axis_ratio=0.95,
                             pa_deg=pa_deg) * bulge_strength

    return disk * (0.5 + 0.7 * arms) + bulge


def _irregular_clumps(size, n_clumps=None, rng=None):
    """Asymmetric clumpy distribution - irregular galaxies."""
    if rng is None:
        rng = np.random.default_rng()
    if n_clumps is None:
        n_clumps = rng.integers(3, 8)

    image = np.zeros((size, size))
    cx, cy = size / 2, size / 2
    for _ in range(n_clumps):
        # Clumps offset from center, but biased toward it
        ox = rng.normal(cx, size * 0.18)
        oy = rng.normal(cy, size * 0.18)
        sigma = rng.uniform(2.0, 5.0)
        amp = rng.uniform(0.4, 1.0)
        y, x = np.indices((size, size))
        clump = amp * np.exp(-((x - ox) ** 2 + (y - oy) ** 2) / (2 * sigma ** 2))
        image += clump
    return image


def _add_noise_and_psf(image, rng, sky_level=0.02, read_noise=0.01,
                        psf_sigma=1.0):
    """Add Gaussian PSF blur, sky background, and read noise."""
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(image, sigma=psf_sigma)
    img = img + sky_level
    # Poisson-like noise (Gaussian approx for speed)
    img = img + rng.normal(0, read_noise, img.shape)
    img = img + rng.normal(0, np.sqrt(np.maximum(img, 0)) * 0.05, img.shape)
    return img


def _normalize(image):
    """Normalize image to [0, 1]."""
    img = image - image.min()
    if img.max() > 0:
        img = img / img.max()
    return img


def make_elliptical(size=IMG_SIZE, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # Lower Sersic indices give visible extended envelope; higher n looks point-like
    n = rng.uniform(1.5, 3.0)
    re = rng.uniform(10.0, 18.0)
    axis_ratio = rng.uniform(0.5, 0.95)
    pa = rng.uniform(0, 180)
    img = _sersic_profile(size, n=n, re=re, axis_ratio=axis_ratio, pa_deg=pa)
    img = _add_noise_and_psf(img, rng, psf_sigma=rng.uniform(1.0, 1.6))
    # Apply a soft saturation to mimic detector response
    # Clip to non-negative before fractional power to avoid NaN from noise
    img = np.power(np.maximum(img, 0), 0.6)
    return _normalize(img)


def make_spiral(size=IMG_SIZE, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    n_arms = rng.choice([2, 3, 4])
    pitch = rng.uniform(12, 28)
    pa = rng.uniform(0, 360)
    disk_scale = rng.uniform(8, 13)
    bulge = rng.uniform(0.3, 0.6)
    img = _spiral_arms(size, n_arms=n_arms, pitch_deg=pitch, pa_deg=pa,
                       disk_scale=disk_scale, bulge_strength=bulge)
    # Apply a viewing inclination (axis ratio < 1 in rare cases)
    img = _add_noise_and_psf(img, rng, psf_sigma=rng.uniform(0.9, 1.3))
    return _normalize(img)


def make_irregular(size=IMG_SIZE, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    img = _irregular_clumps(size, rng=rng)
    img = _add_noise_and_psf(img, rng, psf_sigma=rng.uniform(1.0, 1.6))
    return _normalize(img)


def build_dataset(n_per_class=400, seed=RNG_DEFAULT_SEED):
    """Build a balanced dataset of galaxy images.

    Returns
    -------
    X : array of shape (3 * n_per_class, IMG_SIZE, IMG_SIZE)
    y : array of shape (3 * n_per_class,) with labels 0=Elliptical,
        1=Spiral, 2=Irregular
    """
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    generators = [
        (0, make_elliptical),
        (1, make_spiral),
        (2, make_irregular),
    ]
    for label, gen in generators:
        for _ in range(n_per_class):
            images.append(gen(rng=rng))
            labels.append(label)
    X = np.stack(images, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


CLASS_NAMES = ["Elliptical", "Spiral", "Irregular"]


if __name__ == "__main__":
    X, y = build_dataset(n_per_class=300)
    print(f"Dataset shape: {X.shape}, labels: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    out = Path(__file__).parent / "data" / "galaxy_dataset.npz"
    np.savez_compressed(out, X=X, y=y)
    print(f"Saved to {out}, size = {out.stat().st_size / 1024:.1f} KB")

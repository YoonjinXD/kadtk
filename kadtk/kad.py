import traceback
import warnings
from pathlib import Path
from typing import NamedTuple, Union
import numpy as np
import torch
import shutil
from hypy_utils import write
from hypy_utils.tqdm_utils import tmap, tq

from kadtk.emb_loader import EmbeddingLoader
from kadtk.model_loader import ModelLoader

warnings.filterwarnings("ignore")
PathLike = Union[str, Path]
SCALE_FACTOR = 100

def calc_kernel_audio_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    cache_dirs: tuple[Path, Path],
    device: str,
    bandwidth=None,
    kernel='gaussian',
    precision=torch.float32,
    eps=1e-8
) -> torch.Tensor:
    """
    Compute the Kernel Audio Distance (KAD) between two samples using PyTorch.

    Args:
        x: The first set of embeddings of shape (m, embedding_dim).
        y: The second set of embeddings of shape (n, embedding_dim).
        cache_dirs: Directories to cache kernel statistics.
        bandwidth: The bandwidth value for the Gaussian RBF kernel.
        kernel: Kernel function to use ('gaussian', 'iq', 'imq').
        precision: Type setting for matrix calculation precision.
        eps: Small value to prevent division by zero.

    Returns:
        The KAD between x and y embedding sets.
    """
    # Ensure x and y are of the correct precision
    x = x.to(dtype=precision, device=device)
    y = y.to(dtype=precision, device=device)

    # Use median distance heuristic on the reference set if bandwidth not provided
    if bandwidth is None:
        bandwidth = median_pairwise_distance(x)

    m, n = x.shape[0], y.shape[0]
    
    # Define kernel functions
    gamma = 1 / (2 * bandwidth**2 + eps)
    if kernel == 'gaussian':    # Gaussian Kernel
        kernel = lambda a: torch.exp(-gamma * a)
    elif kernel == 'iq':        # Inverse Quadratic Kernel
        kernel = lambda a: 1 / (1 + gamma * a)
    elif kernel == 'imq':       # Inverse Multiquadric Kernel
        kernel = lambda a: 1 / torch.sqrt(1 + gamma * a)
    else:
        raise ValueError("Invalid kernel type. Valid kernels: 'gaussian', 'iq', 'imq'")
    
    # Load x kernel statistics
    if cache_dirs[0] is not None and cache_dirs[0].exists():
        x_sqnorms = torch.tensor(np.load(cache_dirs[0] / "sqnorms.npy"), dtype=precision, device=device)
        d2_xx = torch.tensor(np.load(cache_dirs[0] / "d2.npy"), dtype=precision, device=device)
    else:
        xx = x @ x.T
        x_sqnorms = torch.diagonal(xx)
        d2_xx = x_sqnorms.unsqueeze(1) + x_sqnorms.unsqueeze(0) - 2 * xx # shape (m, m)
        if cache_dirs[0] is not None:
            cache_dirs[0].mkdir(parents=True, exist_ok=True)
            np.save(cache_dirs[0] / "sqnorms.npy", x_sqnorms.cpu().numpy())
            np.save(cache_dirs[0] / "d2.npy", d2_xx.cpu().numpy())
            
    k_xx = kernel(d2_xx)
    k_xx = k_xx - torch.diag(torch.diagonal(k_xx))
    k_xx_mean = k_xx.sum() / (m * (m - 1))
    
    # Load y kernel statistics
    if cache_dirs[1] is not None and cache_dirs[1].exists():
        y_sqnorms = torch.tensor(np.load(cache_dirs[1] / "sqnorms.npy"), device=device)
        d2_yy = torch.tensor(np.load(cache_dirs[1] / "d2.npy"), device=device)
    else:
        yy = y @ y.T
        y_sqnorms = torch.diagonal(yy)
        d2_yy = y_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * yy # shape (n, n)
        if cache_dirs[1] is not None:
            cache_dirs[1].mkdir(parents=True, exist_ok=True)
            np.save(cache_dirs[1] / "sqnorms.npy", y_sqnorms.cpu().numpy())
            np.save(cache_dirs[1] / "d2.npy", d2_yy.cpu().numpy())

    k_yy = kernel(d2_yy)
    k_yy = k_yy - torch.diag(torch.diagonal(k_yy))
    k_yy_mean = k_yy.sum() / (n * (n - 1))
    
    # Compute kernel statistics for xy
    xy = x @ y.T
    d2_xy = x_sqnorms.unsqueeze(1) + y_sqnorms.unsqueeze(0) - 2 * xy # shape (m, n)
    k_xy = kernel(d2_xy)
    k_xy_mean = k_xy.mean()
    
    # Compute MMD
    result = k_xx_mean + k_yy_mean - 2 * k_xy_mean
    return result * SCALE_FACTOR

def median_pairwise_distance(x, subsample=None):
    """
    Compute the median pairwise distance of an embedding set.
    
    Args:
    x: torch.Tensor of shape (n_samples, embedding_dim)
    subsample: int, number of random pairs to consider (optional)
    
    Returns:
    The median pairwise distance between points in x.
    """
    x = torch.tensor(x, dtype=torch.float32)
    n_samples = x.shape[0]
    
    if subsample is not None and subsample < n_samples * (n_samples - 1) / 2:
        # Randomly select pairs of indices
        idx1 = torch.randint(0, n_samples, (subsample,))
        idx2 = torch.randint(0, n_samples, (subsample,))
        
        # Ensure idx1 != idx2
        mask = idx1 == idx2
        idx2[mask] = (idx2[mask] + 1) % n_samples
        
        # Compute distances for selected pairs
        distances = torch.sqrt(torch.sum((x[idx1] - x[idx2])**2, dim=1))
    else:
        # Compute all pairwise distances
        distances = torch.pdist(x)
        
    return torch.median(distances).item()


class KADInfResults(NamedTuple):
    score: float
    slope: float
    r2: float
    points: list[tuple[int, float]]

class KernelAudioDistance:
    def __init__(self, ml: ModelLoader, device: str, bandwidth: float = None, audio_load_worker: int = 8, logger = None, force_stats_calc=False):
        self.ml = ml
        self.device = torch.device(device)
        self.bandwidth = bandwidth # Bandwidth for the Gaussian kernel
        self.emb_loader = EmbeddingLoader(ml, load_model=False)
        self.audio_load_worker = audio_load_worker
        self.logger = logger
        self.force_stats_calc = force_stats_calc

    def get_cache_dir(self, path: PathLike):
        # Check cache stats
        path = Path(path)
        cache_dir = path / "kernel_stats" / self.ml.name
        if cache_dir.exists(): 
            if self.force_stats_calc:
                self.logger.info(f"Force calculating kernel statistics for {path}.")
                shutil.rmtree(cache_dir)
            else:
                self.logger.info(f"Kernel statistics is already cached for {path}.")
        return cache_dir
    
    def score(self, baseline: PathLike, eval: PathLike):
        """
        Calculate a single KAD score between a background and an eval set.

        :param baseline: Baseline embeddings directory or file
        :param eval: Eval embeddings directory or file
        """
        bg_cache_dir, eval_cache_dir = self.get_cache_dir(baseline), self.get_cache_dir(eval)
        cache_dirs = (bg_cache_dir, eval_cache_dir)
        
        embd_bg = self.emb_loader.load_embeddings(baseline)
        embd_eval = self.emb_loader.load_embeddings(eval)
        
        embd_bg = torch.tensor(embd_bg)
        embd_eval = torch.tensor(embd_eval)
        return calc_kernel_audio_distance(embd_bg, embd_eval, cache_dirs, self.device, self.bandwidth)
    
    def score_inf(self, baseline: PathLike, eval_files: list[Path], steps: int=25, min_n=500, raw: bool=False):
        """
        Calculate KAD for different n (number of samples) and compute KAD-inf.

        :param baseline: Baseline matrix or directory containing baseline audio files
        :param eval_files: list of eval audio files
        :param steps: number of steps to use
        :param min_n: minimum n to use
        :param raw: return raw results in addition to FAD-inf
        :param bandwidth: Bandwidth for the Gaussian kernel
        """
        self.logger.info(f"Calculating KAD-inf for {self.ml.name}...")
        # Load background embeddings
        embd_bg = self.emb_loader.load_embeddings(baseline)
        bg_cache_dir = self.get_cache_dir(baseline)
        cache_dirs = (bg_cache_dir, None)
            
        # If all of the embedding files end in .npy, we can load them directly
        if all([f.suffix == '.npy' for f in eval_files]):
            embeds = [np.load(f) for f in eval_files]
            embeds = np.concatenate(embeds, axis=0)
        else:
            embeds = self.emb_loader._load_embeddings(eval_files, concat=True)
        
        # Calculate maximum n
        max_n = len(embeds)

        # Generate list of ns to use
        ns = [int(n) for n in np.linspace(min_n, max_n, steps)]
        
        results = []
        for n in tq(ns, desc="Calculating KAD-inf"):
            # Select n feature frames randomly (with replacement)
            indices = np.random.choice(embeds.shape[0], size=n, replace=True)
            embds_eval = embeds[indices]

            embd_bg = torch.tensor(embd_bg)
            embds_eval = torch.tensor(embds_eval)
            score = calc_kernel_audio_distance(embd_bg, embds_eval, cache_dirs, self.device, self.bandwidth)
            score = score.item()

            # Add to results
            results.append((n, score))

        # Compute KAD-inf based on linear regression of 1/n
        ys = np.array(results)
        xs = 1 / np.array(ns)
        slope, intercept = np.polyfit(xs, ys[:, 1], 1)

        # Compute R^2
        r2 = 1 - np.sum((ys[:, 1] - (slope * xs + intercept)) ** 2) / np.sum((ys[:, 1] - np.mean(ys[:, 1])) ** 2)

        return KADInfResults(score=intercept, slope=slope, r2=r2, points=results)
    
    def score_individual(self, baseline: PathLike, eval_dir: PathLike, csv_name: Union[Path, str]) -> Path:
        """
        Calculate the KAD score for each individual file in eval_dir and write the results to a csv file.

        :param baseline: Baseline matrix or directory containing baseline audio files
        :param eval_dir: Directory containing eval audio files
        :param csv_name: Name of the csv file to write the results to
        :param bandwidth: Bandwidth for the Gaussian kernel
        :return: Path to the csv file
        """
        if isinstance(csv_name, str):
            csv = Path(csv_name)
            csv = Path('data') / f'result' / self.ml.name / csv_name
            if csv.exists():
                self.logger.info(f"CSV file {csv} already exists, exiting...")
                return csv
        else:
            csv = Path('data') / f'result' / self.ml.name / "kad-indiv.csv"
        
        bg_cache_dir = self.get_cache_dir(baseline)
        cache_dirs = (bg_cache_dir, None)

        # 1. Load background embeddings
        embd_bg = self.emb_loader.load_embeddings(baseline)
        embd_bg = torch.tensor(embd_bg)

        # 2. Define helper function for calculating z score
        def _find_kad_helper(f):
            try:
                # Calculate KAD for individual songs
                embd = self.emb_loader.read_embedding_file(f)
                embd = torch.tensor(embd)
                score = calc_kernel_audio_distance(embd_bg, embd, cache_dirs, self.device, self.bandwidth)
                return score.item()

            except Exception as e:
                traceback.print_exc()
                self.logger.error(f"An error occurred calculating individual KAD using model {self.ml.name} on file {f}")
                self.logger.error(e)
        
        # 3. Calculate MMD score for each eval file
        _files = list(Path(eval_dir).glob("*.*"))
        scores = tmap(_find_kad_helper, _files, desc=f"Calculating scores", max_workers=self.audio_load_worker)

        # 4. Write the sorted scores to csv
        pairs = list(zip(_files, scores))
        pairs = [p for p in pairs if p[1] is not None]
        pairs = sorted(pairs, key=lambda x: np.abs(x[1]))
        write(csv, "\n".join([",".join([str(x).replace(',', '_') for x in row]) for row in pairs]))

        return csv

from ._version import __version__
from .comparison import PermutationResult, permutation_test, subspace_angles
from .core import EventMatrix, decompose
from .dataset import TensorDataset, TensorView, concat_datasets, make_dataset
from .io import from_array, from_edf_batch, from_luna, from_mne, from_npz, from_yasa
from .lwf import from_lwf, lwf_summary, lwf_event_counts, LwfSummary
from .result import DecompositionResult
from .selection import elbow, kaiser, parallel_analysis, select_n_components
from .tensor import AxisAnnotatedTensor
from .tensor_decomp import TensorResult, tensor_decompose

load_result = DecompositionResult.load

__all__ = [
    "__version__",
    # Core API
    "decompose",
    # Data containers
    "AxisAnnotatedTensor",
    "EventMatrix",
    # Dataset / view
    "TensorDataset",
    "TensorView",
    "make_dataset",
    "concat_datasets",
    # I/O
    "from_array",
    "from_npz",
    "from_mne",
    "from_yasa",
    "from_luna",
    "from_edf_batch",
    "from_lwf",
    "lwf_summary",
    "lwf_event_counts",
    "LwfSummary",
    # Component selection
    "elbow",
    "kaiser",
    "parallel_analysis",
    "select_n_components",
    # Comparison
    "subspace_angles",
    "permutation_test",
    "PermutationResult",
    # Serialization
    "DecompositionResult",
    "load_result",
    # Tensor decomposition
    "tensor_decompose",
    "TensorResult",
]

from .dataset_llff import LLFFDataset
from .dataset_omnivideo import OmniVideoDataset
from .dataset_omniblender import OmniBlenderDataset
from .dataset_omniscenes import OmniscenesDataset
from .dataset_omnivideo_cpp import CPP_OmniVideoDataset
from .dataset_omnivideo_stabilization import stabilize_OmniVideoDataset

dataset_dict = {
    'llff': LLFFDataset,
    'omnivideos': OmniVideoDataset,
    'cpp_omnivideos': CPP_OmniVideoDataset,
    'stabilize_omnivideos': stabilize_OmniVideoDataset,
    'omniblender': OmniBlenderDataset,
    'omniscenes': OmniscenesDataset,
}

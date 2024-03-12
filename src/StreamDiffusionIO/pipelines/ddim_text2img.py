
from typing import Dict, Optional

import torch
from diffusers import DDIMScheduler

from .utils import DiffusionStreamIO


class DDIMStreamIO(DiffusionStreamIO):
    
    def __init__(
        self,
        model_id_or_path: str,
        lora_dict: Optional[Dict[str, float]] = None,
        use_xformers: bool = False,
        **base_kwargs,
    ):
        raise NotImplementedError
    
        model_kwargs = {
            "model_id_or_path": model_id_or_path,
            "lora_dict": lora_dict,
            "use_xformers": use_xformers,
        }

        super().__init__(
            model_kwargs,
            **base_kwargs,
        )


    def _load_models(self, **model_kwargs):

        super()._load_models(**model_kwargs)

        self._get_components(scheduler_class=DDIMScheduler)

    @torch.no_grad()
    def _prepare(
        self,
        num_inference_steps: int,
    ):
        raise NotImplementedError
    
    def _scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        sample_batch: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

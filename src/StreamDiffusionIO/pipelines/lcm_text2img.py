from typing import Dict, Optional

import torch
from diffusers import LCMScheduler
from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img import \
    retrieve_timesteps

from .utils import DiffusionStreamIO

torch.set_grad_enabled(False)


class LatentConsistencyModelStreamIO(DiffusionStreamIO):
    
    def __init__(
        self,
        lcm_lora_path: str,
        model_id_or_path: str,
        lora_dict: Optional[Dict[str, float]] = None,
        use_xformers: bool = False,
        **base_kwargs,
    ):
        model_kwargs = {
            "model_id_or_path": model_id_or_path,
            "lcm_lora_path": lcm_lora_path,
            "lora_dict": lora_dict,
            "use_xformers": use_xformers,
        }

        super().__init__(
            model_kwargs,
            **base_kwargs,
        )

    def _load_models(
        self,
        model_id_or_path: str,
        lcm_lora_path: str,
        lora_dict: Optional[Dict[str, float]] = None,
        use_xformers: bool = False,
    ):

        super()._load_models(
            model_id_or_path, 
            lora_dict, 
            use_xformers
        )

        self.pipe.load_lora_weights(lcm_lora_path)
        self.pipe.fuse_lora()
        print(f"Fused LCM-LoRA: {lcm_lora_path}")

        self._get_components(scheduler_class=LCMScheduler)

    @torch.no_grad()
    def _prepare(
        self,
        num_inference_steps: int,
    ):
        self.timesteps, self.num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, self.device
        )

        super()._prepare()

        c_skip_out_list = torch.tensor(
            [
                self.scheduler.get_scalings_for_boundary_condition_discrete(t)
                for t in self.timesteps
            ],
            dtype=self.dtype,
            device=self.device,
        )
        self.c_skip_list = c_skip_out_list[:, 0].view(-1, 1, 1, 1)
        self.c_out_list = c_skip_out_list[:, 1].view(-1, 1, 1, 1)

        alpha_prod_t_list = torch.tensor(
            [self.scheduler.alphas_cumprod[t] for t in self.timesteps],
            dtype=self.dtype,
            device=self.device,
        )
        beta_prod_t_list = 1.0 - alpha_prod_t_list
        self.alpha_prod_t_sqrt_list = torch.sqrt(alpha_prod_t_list).view(-1, 1, 1, 1)
        self.beta_prod_t_sqrt_list = torch.sqrt(beta_prod_t_list).view(-1, 1, 1, 1)

    def _scheduler_step_batch(
        self,
        model_pred_batch: torch.Tensor,
        sample_batch: torch.Tensor,
    ) -> torch.Tensor:
        predicted_original_sample_batch = (
            sample_batch - self.beta_prod_t_sqrt_list * model_pred_batch
        ) / self.alpha_prod_t_sqrt_list
        denoised_batch = (
            self.c_out_list * predicted_original_sample_batch
            + self.c_skip_list * sample_batch
        )

        prev_sample_batch = denoised_batch[:-1]
        self.prev_latent_buffer = (
            self.alpha_prod_t_sqrt_list[1:] * prev_sample_batch
            + self.beta_prod_t_sqrt_list[1:] * self._init_noise[1:]
        )

        return denoised_batch[-1:]

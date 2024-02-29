from queue import Empty, SimpleQueue
from typing import Dict, Literal, Optional

import torch
from diffusers import (AutoencoderKL, LCMScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel)
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img import \
    retrieve_timesteps
from transformers import AutoTokenizer, CLIPTextModel

torch.set_grad_enabled(False)


class LatentConsistencyModelStreamIO:
    def __init__(
        self,
        model_id_or_path: str,
        lcm_lora_path: str,
        lora_dict: Optional[Dict[str, float]] = None,
        num_inference_steps: int = 8,
        resolution: int = 512,
        guidance_scale: float = 8.5,
        device: Literal["cpu", "cuda"] = "cpu",
        dtype: torch.dtype = torch.float32,
        use_xformers: bool = False,
        seed: int = 0,
        output_type: Literal["pt", "pil"] = "pil",
        verbose: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.resolution = resolution
        self.guidance_scale = guidance_scale
        self.output_type = output_type
        self.seed = seed
        self.verbose = verbose

        self._load_models(
            model_id_or_path,
            lcm_lora_path,
            lora_dict,
            use_xformers,
        )

        self.reset(seed)
        self._prepare(num_inference_steps)

    def _load_models(
        self,
        model_id_or_path: str,
        lcm_lora_path: str,
        lora_dict: Optional[Dict[str, float]] = None,
        use_xformers: bool = False,
    ):

        pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            model_id_or_path,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(dtype=self.dtype, device=self.device)

        if use_xformers and torch.cuda.is_available():
            pipe.enable_xformers_memory_efficient_attention()

        if lora_dict is not None:
            for lora_name, scale in lora_dict.items():
                pipe.load_lora_weights(lora_name)
                pipe.fuse_lora(lora_scale=scale)
                print(f"Fused LoRA: {lora_name} with weights {scale}")

        pipe.load_lora_weights(lcm_lora_path)
        pipe.fuse_lora()
        print(f"Fused LCM-LoRA: {lcm_lora_path}")

        self.unet: UNet2DConditionModel = pipe.unet
        self.vae: AutoencoderKL = pipe.vae
        self.tokenizer: AutoTokenizer = pipe.tokenizer
        self.text_encoder: CLIPTextModel = pipe.text_encoder

        self.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        self.vae_scale_factor = pipe.vae_scale_factor
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = seed
        self.generator = torch.Generator(self.device).manual_seed(self.seed)
        self._elapsed_steps = 0
        self._output_image_queue = SimpleQueue()
        self._prompt_queue = SimpleQueue()
        self._remaining_steps = 0

    @torch.no_grad()
    def _prepare(
        self,
        num_inference_steps: int,
    ):

        self.timesteps, self.num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, self.device
        )

        self._latent_shape = (self.unet.config.in_channels,) + (
            self.resolution // self.vae_scale_factor,
        ) * 2
        self._init_noise = self._prepare_latent(self.num_inference_steps)
        self.prev_latent_buffer = self._prepare_latent(self.num_inference_steps - 1)

        self._prompt_embeds_shape = (
            self.num_inference_steps,
            self.tokenizer.model_max_length,
            self.text_encoder.config.hidden_size,
        )
        self.prompt_embeds_batch = torch.zeros(
            self._prompt_embeds_shape, dtype=self.dtype, device=self.device
        )

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

    def _prepare_latent(self, batch_size: int = 1):
        return torch.randn(
            (batch_size,) + self._latent_shape,
            generator=self.generator,
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )

    def _encoder_new_prompt(self, new_prompt: str):
        if new_prompt is not None:
            self._prompt_queue.put(new_prompt)
            self._remaining_steps = self.num_inference_steps

            text_inputs = self.tokenizer(
                new_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            new_prompt_embeds = self.text_encoder(
                text_input_ids.to(self.text_encoder.device)
            ).last_hidden_state

        else:
            new_prompt_embeds = torch.zeros(
                (1, ) + self._prompt_embeds_shape[1:], 
                dtype=self.dtype, device=self.device
            )

        self.prompt_embeds_batch = torch.cat(
            [new_prompt_embeds, self.prompt_embeds_batch[:-1]], dim=0
        )

    def _unet_step_batch(
        self,
        latents_batch: torch.Tensor,
    ) -> torch.Tensor:

        model_pred = self.unet(
            latents_batch,
            self.timesteps,
            encoder_hidden_states=self.prompt_embeds_batch,
            return_dict=False,
        )[0]

        return model_pred

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

    def _step_batch(self) -> torch.Tensor:
        latent = self._prepare_latent()
        latents_batch = torch.cat((latent, self.prev_latent_buffer), dim=0)
        model_pred_batch = self._unet_step_batch(latents_batch)
        denoised_x0 = self._scheduler_step_batch(model_pred_batch, latents_batch)

        image = self.vae.decode(
            denoised_x0 / self.vae.config.scaling_factor, return_dict=False
        )[0]
        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(
            image, output_type=self.output_type, do_denormalize=do_denormalize
        )[0]

        self._elapsed_steps += 1
        self._remaining_steps -= 1
        if self._elapsed_steps >= self.num_inference_steps:
            self._output_image_queue.put(image)

    def _get_image_and_prompt(self):
        try:
            image = self._output_image_queue.get_nowait()
            return image, self._prompt_queue.get_nowait()
        except Empty:
            if self.verbose:
                print("No image available yet. Returning None.")
            return None, None
    
    def stop(self):
        return (self._remaining_steps <= 0)

    @torch.no_grad()
    def __call__(self, prompt: Optional[str]):

        self._encoder_new_prompt(prompt)
        self._step_batch()

        return self._get_image_and_prompt()

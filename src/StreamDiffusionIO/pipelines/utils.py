from queue import Empty, SimpleQueue
from typing import Dict, Literal, Optional

import torch
from diffusers import (AutoencoderKL, StableDiffusionPipeline,
                       UNet2DConditionModel)
from diffusers.configuration_utils import ConfigMixin
from diffusers.image_processor import VaeImageProcessor
from transformers import AutoTokenizer, CLIPTextModel


class DiffusionStreamIO:

    def __init__(
        self,
        model_kwargs: dict,
        # below are base_kwargs
        num_inference_steps: int = 8,
        resolution: int = 512,
        guidance_scale: float = 8.5,
        device: Literal["cpu", "cuda"] = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
        output_type: Literal["pt", "pil"] = "pil",
        verbose: bool = False,
    ) -> None:

        self.device = device
        self.dtype = dtype
        self.resolution = resolution
        self.guidance_scale = guidance_scale
        self.output_type = output_type
        self.seed = seed
        self.verbose = verbose

        self.reset(seed)

        self._load_models(**model_kwargs)

        self._prepare(num_inference_steps)

    def _load_models(
        self,
        model_id_or_path: str,
        lora_dict: Optional[Dict[str, float]] = None,
        use_xformers: bool = False,
    ):
        self.pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
            model_id_or_path,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(dtype=self.dtype, device=self.device)

        if use_xformers and torch.cuda.is_available():
            self.pipe.enable_xformers_memory_efficient_attention()

        if lora_dict is not None:
            for lora_name, scale in lora_dict.items():
                self.pipe.load_lora_weights(lora_name)
                self.pipe.fuse_lora(lora_scale=scale)
                print(f"Fused LoRA: {lora_name} with weights {scale}")

    def _get_components(self, scheduler_class: ConfigMixin):
        self.unet: UNet2DConditionModel = self.pipe.unet
        self.vae: AutoencoderKL = self.pipe.vae
        self.tokenizer: AutoTokenizer = self.pipe.tokenizer
        self.text_encoder: CLIPTextModel = self.pipe.text_encoder

        self.scheduler = scheduler_class.from_config(self.pipe.scheduler.config)

        self.vae_scale_factor = self.pipe.vae_scale_factor
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
    ):
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
    ) -> torch.Tensor:
        raise NotImplementedError
    
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
    
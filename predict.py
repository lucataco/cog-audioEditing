# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import utils
import torch
import random
import torchaudio
import numpy as np
from models import load_model
from torch import inference_mode
from tempfile import NamedTemporaryFile
from inversion_utils import inversion_forward_process, inversion_reverse_process

LDM2 = "cvssp/audioldm2"
MUSIC = "cvssp/audioldm2-music"
LDM2_LARGE = "cvssp/audioldm2-large"

def randomize_seed_fn(seed, randomize_seed):
    if randomize_seed:
        seed = random.randint(0, np.iinfo(np.int32).max)
    torch.manual_seed(seed)
    return seed

def invert(ldm_stable, x0, prompt_src, num_diffusion_steps, cfg_scale_src):
    ldm_stable.model.scheduler.set_timesteps(num_diffusion_steps, device='cuda')

    with inference_mode():
        w0 = ldm_stable.vae_encode(x0)

    # find Zs and wts - forward process
    _, zs, wts = inversion_forward_process(ldm_stable, w0, etas=1,
                                           prompts=[prompt_src],
                                           cfg_scales=[cfg_scale_src],
                                           prog_bar=True,
                                           num_inference_steps=num_diffusion_steps,
                                           numerical_fix=True)
    return zs, wts

def sample(ldm_stable, zs, wts, steps, prompt_tar, tstart, cfg_scale_tar):
    # reverse process (via Zs and wT)
    tstart = torch.tensor(tstart, dtype=torch.int)
    skip = steps - tstart
    w0, _ = inversion_reverse_process(ldm_stable, xT=wts, skips=steps - skip,
                                      etas=1., prompts=[prompt_tar],
                                      neg_prompts=[""], cfg_scales=[cfg_scale_tar],
                                      prog_bar=True,
                                      zs=zs[:int(steps - skip)])

    # vae decode image
    with inference_mode():
        x0_dec = ldm_stable.vae_decode(w0)
    if x0_dec.dim() < 4:
        x0_dec = x0_dec[None, :, :, :]

    with torch.no_grad():
        audio = ldm_stable.decode_to_mel(x0_dec)

    f = NamedTemporaryFile("wb", suffix=".wav", delete=False)
    torchaudio.save(f.name, audio, sample_rate=16000)

    return f.name

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ldm2 = load_model(model_id=LDM2, device=self.device)
        self.ldm2_large = load_model(model_id=LDM2_LARGE, device=self.device)
        self.ldm2_music = load_model(model_id=MUSIC, device=self.device)

    def predict(
        self,
        audio: Path = Input(description="Input Audio File"),
        prompt: str = Input(
            description="Describe your desired edited output",
            default="A recording of an arcade game soundtrack"
        ),
        t_start: int = Input(
            description="Lower % returns closer to the original audio, higher returns stronger edit",
            default=45, ge=15, le=85
        ),
        audio_version: str = Input(
            description="Choose the audio version to return",
            default="cvssp/audioldm2-music",
            choices=["cvssp/audioldm2", "cvssp/audioldm2-large", "cvssp/audioldm2-music"]
        ),
        source_prompt: str = Input(description="Optional: describe the original audio input", default=""),
        steps: int = Input(
            description="Number of diffusion steps, higher values(200) yield high-quality generations",
            default=50
        ),
        cfg_scale_src: float = Input(description="Source Guidance Scale", default=3.0),
        cfg_scale_tar: float = Input(description="Target Guidance Scale", default=12.0),
        seed: int = Input(description="Random seed", default=None),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"Using seed: {seed}")
        torch.Generator("cuda").manual_seed(seed)

        input_audio = str(audio)
        model_id = audio_version

        print("Using model:", model_id)
        if model_id == LDM2:
            ldm_stable = self.ldm2
        elif model_id == LDM2_LARGE:
            ldm_stable = self.ldm2_large
        else:
            ldm_stable = self.ldm2_music

        x0 = utils.load_audio(input_audio, ldm_stable.get_fn_STFT(), device=self.device)

        zs_tensor, wts_tensor = invert(ldm_stable=ldm_stable, x0=x0, prompt_src=source_prompt,
                                    num_diffusion_steps=steps,
                                    cfg_scale_src=cfg_scale_src)
        wts = wts_tensor
        zs = zs_tensor

        output = sample(ldm_stable, zs, wts, steps, prompt_tar=prompt,
                        tstart=int(t_start / 100 * steps), cfg_scale_tar=cfg_scale_tar)

        return Path(output)
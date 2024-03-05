import numpy as np
import torch
from typing import Optional, List, Tuple, NamedTuple, Union
from models import PipelineWrapper


class PromptEmbeddings(NamedTuple):
    embedding_hidden_states: torch.Tensor
    embedding_class_lables: torch.Tensor
    boolean_prompt_mask: torch.Tensor


def load_audio(audio_path: Union[str, np.array], fn_STFT, left: int = 0, right: int = 0, device: Optional[torch.device] = None
               ) -> torch.tensor:
    if type(audio_path) is str:
        import audioldm
        import audioldm.audio

        duration = min(audioldm.utils.get_duration(audio_path), 30)

        mel, _, _ = audioldm.audio.wav_to_fbank(audio_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT)
        mel = mel.unsqueeze(0)
    else:
        mel = audio_path

    c, h, w = mel.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    mel = mel[:, :, left:w-right]
    mel = mel.unsqueeze(0).to(device)

    return mel


def get_height_of_spectrogram(length: int, ldm_stable: PipelineWrapper) -> int:
    vocoder_upsample_factor = np.prod(ldm_stable.model.vocoder.config.upsample_rates) / \
        ldm_stable.model.vocoder.config.sampling_rate

    if length is None:
        length = ldm_stable.model.unet.config.sample_size * ldm_stable.model.vae_scale_factor * \
            vocoder_upsample_factor

    height = int(length / vocoder_upsample_factor)

    # original_waveform_length = int(length * ldm_stable.model.vocoder.config.sampling_rate)
    if height % ldm_stable.model.vae_scale_factor != 0:
        height = int(np.ceil(height / ldm_stable.model.vae_scale_factor)) * ldm_stable.model.vae_scale_factor
        print(
            f"Audio length in seconds {length} is increased to {height * vocoder_upsample_factor} "
            f"so that it can be handled by the model. It will be cut to {length} after the "
            f"denoising process."
        )

    return height


def get_text_embeddings(target_prompt: List[str], target_neg_prompt: List[str], ldm_stable: PipelineWrapper
                        ) -> Tuple[torch.Tensor, PromptEmbeddings, PromptEmbeddings]:
    text_embeddings_hidden_states, text_embeddings_class_labels, text_embeddings_boolean_prompt_mask = \
        ldm_stable.encode_text(target_prompt)
    uncond_embedding_hidden_states, uncond_embedding_class_lables, uncond_boolean_prompt_mask = \
        ldm_stable.encode_text(target_neg_prompt)

    text_emb = PromptEmbeddings(embedding_hidden_states=text_embeddings_hidden_states,
                                boolean_prompt_mask=text_embeddings_boolean_prompt_mask,
                                embedding_class_lables=text_embeddings_class_labels)
    uncond_emb = PromptEmbeddings(embedding_hidden_states=uncond_embedding_hidden_states,
                                  boolean_prompt_mask=uncond_boolean_prompt_mask,
                                  embedding_class_lables=uncond_embedding_class_lables)

    return text_embeddings_class_labels, text_emb, uncond_emb
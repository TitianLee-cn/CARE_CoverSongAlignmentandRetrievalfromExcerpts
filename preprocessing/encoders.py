"""Adapters for CARE's three frozen audio encoders."""

import sys

import numpy as np
import torch

from common import CARE, DATA, source_version, weight_versions


class WindowEncoder:
    def encode(self, audio):
        vectors, frame_count = self.encode_batch(audio[None, :])
        return vectors[0], frame_count


class Mert(WindowEncoder):
    sample_rate = 24000
    dimension = 768
    default_batch_size = 4

    def __init__(self, device):
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.device = device
        path = DATA / "weights" / "mert-v1-95m"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            path, trust_remote_code=True, local_files_only=True
        ).eval().to(device)
        self.version = {
            "weights": weight_versions(*(path / name for name in (
                "pytorch_model.bin", "config.json", "preprocessor_config.json",
                "modeling_MERT.py", "configuration_MERT.py",
            ))),
            "frontend": "Wav2Vec2FeatureExtractor",
        }

    @torch.inference_mode()
    def frames(self, audio):
        inputs = self.feature_extractor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt",
        ).to(self.device)
        return self.model(**inputs).last_hidden_state[0].float().cpu().numpy()

    def frame_geometry(self):
        stride, width = 1, 1
        config = self.model.config
        for kernel, step in zip(config.conv_kernel, config.conv_stride):
            width += (kernel - 1) * stride
            stride *= step
        return stride, width

    @torch.inference_mode()
    def encode_batch(self, audio):
        inputs = self.feature_extractor(
            list(audio), sampling_rate=self.sample_rate, return_tensors="pt",
        ).to(self.device)
        frames = self.model(**inputs).last_hidden_state
        return frames.float().mean(dim=1).cpu().numpy(), frames.shape[1]


class CoverHunter(WindowEncoder):
    sample_rate = 16000
    dimension = 128
    default_batch_size = 8

    def __init__(self, device):
        if device.type != "cuda":
            raise ValueError("The supplied CoverHunter implementation requires CUDA")
        root = CARE / "vendor" / "CoverHunter"
        sys.path.insert(0, str(root))
        from src.cqt import PyCqt
        from src.model import Model
        from src.utils import load_hparams

        self.device = device
        path = DATA / "weights" / "coverhunter" / "pretrain_model"
        config_path = path / "config" / "hparams.yaml"
        weight = path / "pt_model" / "g_00000040"
        config = load_hparams(str(config_path))
        self.mean_size = config["mean_size"]
        hop_seconds = config["cqt"]["hop_size"]
        self.hop = round(self.sample_rate / int(1 / hop_seconds))
        # Use the exact frequency-domain filter bank from the released frontend.
        kernel = PyCqt(self.sample_rate, hop_seconds)._kernel.tocoo()
        indices = np.stack((kernel.row, kernel.col))
        self.kernel = torch.sparse_coo_tensor(
            indices, kernel.data.astype(np.complex64), kernel.shape, device=device, check_invariants=True,
        ).coalesce()
        self.model = Model(config).to(device)
        self.model.load_model_parameters(str(weight.parent), epoch_num=40, device=str(device))
        self.version = {
            "source_commit": source_version(root), "weights": weight_versions(weight, config_path),
            "frontend": "PyCqt", "cqt_hop_sec": config["cqt"]["hop_size"], "mean_size": self.mean_size,
        }

    @torch.inference_mode()
    def encode_batch(self, audio):
        waveform = torch.as_tensor(audio, device=self.device)
        peak = waveform.abs().amax(dim=1, keepdim=True).clamp_min(0.001)
        waveform = waveform / peak * 0.999
        fft_size = self.kernel.shape[1]
        pad = fft_size - self.hop
        padded = torch.nn.functional.pad(waveform, ((pad + 1) // 2, pad // 2))
        frames = padded.unfold(1, fft_size, self.hop)
        spectrum = torch.fft.fft(frames.contiguous(), dim=-1)
        columns = spectrum.reshape(-1, fft_size).T.contiguous()
        magnitude = torch.sparse.mm(self.kernel, columns).abs().T
        cqt = magnitude.reshape(len(audio), frames.shape[1], -1).add_(1e-9).log10_()
        cqt = 20 * (cqt - cqt.amax(dim=(1, 2), keepdim=True))
        valid = cqt.shape[1] // self.mean_size
        features = cqt[:, :valid * self.mean_size].reshape(
            len(audio), valid, self.mean_size, -1,
        ).mean(dim=2)
        vectors, _ = self.model.inference(features)
        return vectors.float().cpu().numpy(), valid


class Unified(WindowEncoder):
    sample_rate = 16000
    dimension = 1024
    default_batch_size = 8

    def __init__(self, device):
        root = CARE / "vendor" / "unified-track-and-version-id"
        sys.path.insert(0, str(root))
        from src.fish.model import FishModule

        self.device = device
        path = DATA / "weights" / "unified-music-identification" / "epoch-8-step-90000.ckpt"
        self.model = FishModule.load_from_checkpoint(path, map_location="cpu").eval().to(device)
        self.model.feature_processor.register_forward_hook(self._record_frame_count)
        self.version = {
            "source_commit": source_version(root), "weights": weight_versions(path),
            "frontend": "FishModule.extract_embeddings", "layer": "proj",
        }

    def _record_frame_count(self, module, inputs, output):
        self.frame_count = output.shape[-1]

    @torch.inference_mode()
    def encode_batch(self, audio):
        inputs = torch.from_numpy(audio).unsqueeze(1).to(self.device)
        samples = audio.shape[1]
        vectors = self.model.extract_embeddings(
            inputs, true_lengths=[samples] * len(audio),
            segment_duration=samples / self.sample_rate, overlap_ratio=0,
            output_half=False, layer="proj",
        )
        if any(vector.shape != (1, self.dimension) for vector in vectors):
            raise ValueError("Unified returned an unexpected number of embeddings per window")
        return torch.cat(vectors).float().cpu().numpy(), self.frame_count

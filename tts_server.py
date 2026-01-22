import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict
import io

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download
from lhotse.utils import fix_random_seed
from vocos import Vocos

from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.tokenizer import EspeakTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import (
    add_punctuation,
    batchify_tokens,
    chunk_tokens_punctuation,
    cross_fade_concat,
    fade_in_out,
    load_prompt_wav,
    remove_silence,
    rms_norm,
)

import dotenv

dotenv.load_dotenv(override = "True")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HUGGINGFACE_REPO = "hieuclc/zipVoice-vietnamese-1000h"

import torchaudio.functional as F

def resample_24000_to_16000(wav: torch.Tensor) -> torch.Tensor:
    """
    wav: Tensor shape (T,) hoặc (1, T)
    return: Tensor shape (1, T_new)
    """
    wav = wav.squeeze()  # (T,)
    wav = F.resample(
        wav,
        orig_freq=24000,
        new_freq=16000
    )
    return wav.unsqueeze(0)  # (1, T_new)


# Request/Response models
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice_id: str = Field(default="default", description="Voice ID to use (must be pre-cached)")
    num_step: int = Field(default=16, ge=1, le=100, description="Number of diffusion steps")
    guidance_scale: float = Field(default=1.0, ge=0.0, le=10.0, description="Guidance scale for generation")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    t_shift: float = Field(default=0.5, ge=0.0, le=1.0, description="Time shift parameter")
    target_rms: float = Field(default=0.1, ge=0.0, le=1.0, description="Target RMS volume")
    remove_long_sil: bool = Field(default=False, description="Remove long silences from output")

class VoiceMetrics(BaseModel):
    total_time: float
    model_time: float
    vocoder_time: float
    audio_duration: float
    rtf: float
    rtf_no_vocoder: float
    rtf_vocoder: float


class VoiceCache:
    """Cache for preprocessed voice prompts"""
    
    def __init__(self):
        self.prompt_text: str = ""
        self.prompt_wav: torch.Tensor = None
        self.prompt_features: torch.Tensor = None
        self.prompt_tokens: list = None
        self.prompt_duration: float = 0.0
        self.prompt_rms: float = 0.0


class TTSEngine:
    """TTS Engine that handles model initialization and inference"""
    
    def __init__(
        self,
        model_name: str = "zipvoice",
        checkpoint_name: str = "iter-10000-avg-2.pt",
        model_config_file: str = "config.json",
        model_token_file: str = "tokens.txt",
        lang: str = "vi",
        model_dir: Optional[str] = None,
        vocoder_path: Optional[str] = None,
        trt_engine_path: Optional[str] = None,
        device: Optional[str] = None,
        seed: int = 42,
        voice_configs: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        """Initialize TTS engine with all required components"""
        
        logger.info("Initializing TTS Engine...")
        
        self.params = AttributeDict()
        self.params.model_name = model_name
        self.params.checkpoint_name = checkpoint_name
        self.params.model_config_file = model_config_file
        self.params.model_token_file = model_token_file
        self.params.lang = lang
        self.params.model_dir = Path(model_dir) if model_dir else None
        self.params.vocoder_path = vocoder_path
        self.params.trt_engine_path = trt_engine_path
        self.params.seed = seed
        
        fix_random_seed(seed)
        
        # Load model files
        self._load_model_files()
        
        # Initialize tokenizer
        self.tokenizer = EspeakTokenizer(
            token_file=self.token_file,
            lang=self.params.lang
        )
        
        # Load model config
        with open(self.model_config, "r") as f:
            model_config = json.load(f)
        
        self.model_config_dict = model_config
        
        # Initialize model
        tokenizer_config = {
            "vocab_size": self.tokenizer.vocab_size,
            "pad_id": self.tokenizer.pad_id
        }
        
        self.model = ZipVoice(
            **model_config["model"],
            **tokenizer_config,
        )
        
        load_checkpoint(filename=self.model_ckpt, model=self.model, strict=True)
        
        # Setup device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"Using device: {self.device}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load TensorRT if specified
        if self.params.trt_engine_path:
            from zipvoice.utils.tensorrt import load_trt
            load_trt(self.model, self.params.trt_engine_path)
        
        # Initialize vocoder
        self.vocoder = self._get_vocoder()
        self.vocoder = self.vocoder.to(self.device)
        self.vocoder.eval()
        
        # Initialize feature extractor
        self.feature_extractor = VocosFbank()
        self.sampling_rate = model_config["feature"]["sampling_rate"]
        
        # Initialize voice cache
        self.voice_cache: Dict[str, VoiceCache] = {}
        
        # Cache voices if config provided
        if voice_configs:
            self._cache_voices(voice_configs)
        
        logger.info("TTS Engine initialized successfully!")
    
    def _load_model_files(self):
        """Load model checkpoint and config files"""
        if self.params.model_dir and self.params.model_dir.is_dir():
            # Load from local directory
            self.model_ckpt = self.params.model_dir / self.params.checkpoint_name
            self.model_config = self.params.model_dir / self.params.model_config_file
            self.token_file = self.params.model_dir / self.params.model_token_file
            
            for filepath in [self.model_ckpt, self.model_config, self.token_file]:
                if not filepath.is_file():
                    raise FileNotFoundError(f"{filepath} does not exist")
            
            logger.info(f"Using local model from {self.params.model_dir}")
        else:
            # Download from HuggingFace
            logger.info(f"Downloading model from HuggingFace: {HUGGINGFACE_REPO}")
            
            self.model_ckpt = hf_hub_download(
                HUGGINGFACE_REPO,
                filename=self.params.checkpoint_name
            )
            self.model_config = hf_hub_download(
                HUGGINGFACE_REPO,
                filename=self.params.model_config_file
            )
            self.token_file = hf_hub_download(
                HUGGINGFACE_REPO,
                filename=self.params.model_token_file
            )
    
    def _get_vocoder(self):
        """Initialize vocoder"""
        if self.params.vocoder_path:
            vocoder = Vocos.from_hparams(f"{self.params.vocoder_path}/config.yaml")
            state_dict = torch.load(
                f"{self.params.vocoder_path}/pytorch_model.bin",
                weights_only=True,
                map_location="cpu",
            )
            vocoder.load_state_dict(state_dict)
        else:
            vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        return vocoder
    
    def _cache_voices(self, voice_configs: Dict[str, Dict[str, str]]):
        """
        Cache voice prompts during initialization
        
        Args:
            voice_configs: Dict mapping voice_id to {"prompt_text": str, "prompt_wav": str}
        """
        logger.info("Caching voices...")
        
        for voice_id, config in voice_configs.items():
            try:
                self.cache_voice(
                    voice_id=voice_id,
                    prompt_text=config["prompt_text"],
                    prompt_wav=config["prompt_wav"],
                    target_rms=config.get("target_rms", 0.1),
                    feat_scale=config.get("feat_scale", 0.1),
                )
                logger.info(f"Cached voice: {voice_id}")
            except Exception as e:
                logger.error(f"Failed to cache voice {voice_id}: {str(e)}")
        
        logger.info(f"Voice caching complete. Total voices: {len(self.voice_cache)}")
    
    def cache_voice(
        self,
        voice_id: str,
        prompt_text: str,
        prompt_wav: str,
        target_rms: float = 0.1,
        feat_scale: float = 0.1,
    ):
        """
        Cache a voice prompt for faster inference
        
        Args:
            voice_id: Unique identifier for this voice
            prompt_text: Reference text for voice cloning
            prompt_wav: Path to prompt audio file
            target_rms: Target RMS volume
            feat_scale: Feature scaling factor
        """
        cache = VoiceCache()
        
        # Load and process prompt wav
        prompt_wav_tensor = load_prompt_wav(prompt_wav, sampling_rate=self.sampling_rate)
        
        # Remove edge and long silences
        prompt_wav_tensor = remove_silence(
            prompt_wav_tensor, self.sampling_rate, only_edge=False, trail_sil=200
        )
        
        prompt_wav_tensor, prompt_rms = rms_norm(prompt_wav_tensor, target_rms)
        prompt_duration = prompt_wav_tensor.shape[-1] / self.sampling_rate
        
        if prompt_duration > 20:
            logger.warning(
                f"Prompt audio for {voice_id} is too long ({prompt_duration}s). "
                f"Recommended: 1-3 seconds."
            )
        elif prompt_duration > 10:
            logger.warning(
                f"Prompt audio for {voice_id} is long ({prompt_duration}s)."
            )
        
        # Extract features from prompt wav
        prompt_features = self.feature_extractor.extract(
            prompt_wav_tensor, sampling_rate=self.sampling_rate
        ).to(self.device)
        
        prompt_features = prompt_features.unsqueeze(0) * feat_scale
        
        # Add punctuation and tokenize
        prompt_text = add_punctuation(prompt_text)
        prompt_tokens_str = self.tokenizer.texts_to_tokens([prompt_text])[0]
        prompt_tokens = self.tokenizer.tokens_to_token_ids([prompt_tokens_str])
        
        # Store in cache
        cache.prompt_text = prompt_text
        cache.prompt_wav = prompt_wav_tensor
        cache.prompt_features = prompt_features
        cache.prompt_tokens = prompt_tokens
        cache.prompt_duration = prompt_duration
        cache.prompt_rms = prompt_rms
        
        self.voice_cache[voice_id] = cache
        
        logger.info(
            f"Voice {voice_id} cached successfully. "
            f"Duration: {prompt_duration:.2f}s"
        )
    
    @torch.inference_mode()
    def infer(
        self,
        text: str,
        voice_id: str = "default",
        num_step: int = 16,
        guidance_scale: float = 1.0,
        speed: float = 1.0,
        t_shift: float = 0.5,
        target_rms: float = 0.1,
        feat_scale: float = 0.1,
        max_duration: float = 100,
        remove_long_sil: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Perform TTS inference using cached voice
        
        Args:
            text: Text to synthesize
            voice_id: ID of cached voice to use
            Other parameters control generation quality and speed
        
        Returns:
            Tuple of (audio_tensor, metrics_dict)
        """
        
        # Get cached voice
        if voice_id not in self.voice_cache:
            raise ValueError(
                f"Voice ID '{voice_id}' not found in cache. "
                f"Available voices: {list(self.voice_cache.keys())}"
            )
        
        cache = self.voice_cache[voice_id]
        
        # Add punctuation to text
        text = add_punctuation(text)
        
        # Tokenize text
        tokens_str = self.tokenizer.texts_to_tokens([text])[0]
        
        # Chunk text
        token_duration = cache.prompt_duration / (len(cache.prompt_tokens[0]) * speed)
        max_tokens = int((25 - cache.prompt_duration) / token_duration)
        chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)
        
        # Tokenize to int tokens
        chunked_tokens = self.tokenizer.tokens_to_token_ids(chunked_tokens_str)
        
        # Batchify tokens
        tokens_batches, chunked_index = batchify_tokens(
            chunked_tokens, max_duration, cache.prompt_duration, token_duration
        )
        
        # Generate features
        chunked_features = []
        start_t = dt.datetime.now()
        
        for batch_tokens in tokens_batches:
            batch_prompt_tokens = cache.prompt_tokens * len(batch_tokens)
            batch_prompt_features = cache.prompt_features.repeat(len(batch_tokens), 1, 1)
            batch_prompt_features_lens = torch.full(
                (len(batch_tokens),), cache.prompt_features.size(1), device=self.device
            )
            
            # Generate features
            (
                pred_features,
                pred_features_lens,
                pred_prompt_features,
                pred_prompt_features_lens,
            ) = self.model.sample(
                tokens=batch_tokens,
                prompt_tokens=batch_prompt_tokens,
                prompt_features=batch_prompt_features,
                prompt_features_lens=batch_prompt_features_lens,
                speed=speed,
                t_shift=t_shift,
                duration="predict",
                num_step=num_step,
                guidance_scale=guidance_scale,
            )
            
            # Postprocess predicted features
            pred_features = pred_features.permute(0, 2, 1) / feat_scale
            chunked_features.append((pred_features, pred_features_lens))
        
        # Vocoder processing
        chunked_wavs = []
        start_vocoder_t = dt.datetime.now()
        
        for pred_features, pred_features_lens in chunked_features:
            batch_wav = []
            for i in range(pred_features.size(0)):
                wav = (
                    self.vocoder.decode(pred_features[i][None, :, : pred_features_lens[i]])
                    .squeeze(1)
                    .clamp(-1, 1)
                )
                # Adjust wav volume if necessary
                if cache.prompt_rms < target_rms:
                    wav = wav * cache.prompt_rms / target_rms
                batch_wav.append(wav)
            chunked_wavs.extend(batch_wav)
        
        # Calculate timing
        t = (dt.datetime.now() - start_t).total_seconds()
        
        # Merge chunked wavs
        indexed_chunked_wavs = [
            (index, wav) for index, wav in zip(chunked_index, chunked_wavs)
        ]
        sequential_indexed_chunked_wavs = sorted(indexed_chunked_wavs, key=lambda x: x[0])
        sequential_chunked_wavs = [
            sequential_indexed_chunked_wavs[i][1]
            for i in range(len(sequential_indexed_chunked_wavs))
        ]
        final_wav = cross_fade_concat(
            sequential_chunked_wavs, fade_duration=0.1, sample_rate=self.sampling_rate
        )
        final_wav = remove_silence(
            final_wav, self.sampling_rate, only_edge=(not remove_long_sil), trail_sil=0.2
        )
        # 🔽 RESAMPLE 24k → 16k
        # final_wav = resample_24000_to_16000(final_wav)

        # final_wav = fade_in_out(final_wav, sample_rate = 24000, fade_in_sec=0.2, fade_out_sec=0.2)
        
        # Calculate metrics
        t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
        t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
        wav_seconds = final_wav.shape[-1] / self.sampling_rate
        rtf = t / wav_seconds
        rtf_no_vocoder = t_no_vocoder / wav_seconds
        rtf_vocoder = t_vocoder / wav_seconds
        
        metrics = {
            "total_time": t,
            "model_time": t_no_vocoder,
            "vocoder_time": t_vocoder,
            "audio_duration": wav_seconds,
            "rtf": rtf,
            "rtf_no_vocoder": rtf_no_vocoder,
            "rtf_vocoder": rtf_vocoder,
        }
        
        logger.info(f"Generated audio: {wav_seconds:.2f}s, RTF: {rtf:.3f}")
        
        return final_wav.cpu(), metrics


# Initialize FastAPI app
app = FastAPI(
    title="ZipVoice TTS API",
    description="Vietnamese Text-to-Speech API using ZipVoice with voice caching",
    version="2.0.0"
)

# Global TTS engine instance
tts_engine: Optional[TTSEngine] = None


@app.on_event("startup")
async def startup_event():
    """Initialize TTS engine on startup"""
    global tts_engine
    
    # Define voice configurations
    voice_configs = {
        "alloy": {
            "prompt_text": "bạn đã sẵn sàng cho lễ hội sôi động nhất thái lan?",
            "prompt_wav": "./prompt_voice.mp3",
            "target_rms": 0.1,
            "feat_scale": 0.1,
        },
    }
    
    # Initialize TTS engine with voice caching
    tts_engine = TTSEngine(
        model_name="zipvoice",
        checkpoint_name="iter-10000-avg-2.pt",
        model_config_file="config.json",
        model_token_file="tokens.txt",
        lang="vi",
        model_dir=os.getenv("MODEL_DIR"),
        vocoder_path=os.getenv("VOCODER_PATH"),
        device=os.getenv("DEVICE"),
        seed=42,
        voice_configs=voice_configs,  # Pass voice configs for caching
    )
    
    logger.info("TTS API is ready!")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "ZipVoice TTS API is running",
        "engine_ready": tts_engine is not None,
        "cached_voices": list(tts_engine.voice_cache.keys()) if tts_engine else []
    }


@app.get("/voices")
async def list_voices():
    """List all cached voices"""
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not initialized")
    
    voices = {}
    for voice_id, cache in tts_engine.voice_cache.items():
        voices[voice_id] = {
            "prompt_text": cache.prompt_text,
            "duration": cache.prompt_duration,
        }
    
    return {"voices": voices}


@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    """
    Synthesize speech from text using cached voice
    
    Args:
        request: TTSRequest containing text and generation parameters
    
    Returns:
        WAV audio file as streaming response with metrics in headers
    """
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not initialized")
    
    try:
        # Perform inference
        audio_tensor, metrics = tts_engine.infer(
            text=request.text,
            voice_id=request.voice_id,
            num_step=request.num_step,
            guidance_scale=request.guidance_scale,
            speed=request.speed,
            t_shift=request.t_shift,
            target_rms=request.target_rms,
            remove_long_sil=request.remove_long_sil,
        )
        
        # Convert tensor to WAV bytes
        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            audio_tensor,
            sample_rate=tts_engine.sampling_rate,
            format="wav"
        )
        buffer.seek(0)
        
        # Return audio with metrics in headers
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "X-Audio-Duration": str(metrics["audio_duration"]),
                "X-Total-Time": str(metrics["total_time"]),
                "X-RTF": str(metrics["rtf"]),
                "X-Model-Time": str(metrics["model_time"]),
                "X-Vocoder-Time": str(metrics["vocoder_time"]),
                "Content-Disposition": f'attachment; filename="synthesized.wav"'
            }
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during synthesis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
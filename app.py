"""
app.py — Minimal Text-to-Video Telegram Bot (CPU-only)
Includes monkey-patches for:
  • huggingface_hub.cached_download
  • jax.random.KeyArray
to satisfy diffusers v0.12.1 on Colab.
"""

# 1) Patch huggingface_hub so diffusers can use cached_download
import huggingface_hub
if not hasattr(huggingface_hub, "cached_download"):
    from huggingface_hub import hf_hub_download
    huggingface_hub.cached_download = hf_hub_download

# 2) Patch JAX so FlaxModelMixin sees KeyArray
try:
    import jax
    jax.random.KeyArray = jax.random.PRNGKey
except ImportError:
    pass

import yaml
import time
import logging
from functools import wraps

import torch
from diffusers import DiffusionPipeline
from telegram import Bot, Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from moviepy.editor import ImageSequenceClip
import cv2

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Retry decorator
def retry(exceptions, tries=3, delay=2):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logging.warning(f"Attempt {attempt} failed: {e}")
                    time.sleep(delay)
            return func(*args, **kwargs)
        return wrapper
    return deco

# Load configuration from config.yaml
@retry(Exception)
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Load Diffusers pipeline on CPU
@retry(Exception)
def load_pipeline(model_id, revision=None, torch_dtype=torch.float32):
    device = "cpu"
    logging.info(f"Loading model '{model_id}' on {device}")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        safety_checker=None,
        device_map=device
    )
    pipe.enable_attention_slicing()
    return pipe

# Generate video frames from text prompt
@retry(Exception)
def generate_frames(pipe, prompt, num_frames, height, width, guidance_scale, num_inference_steps):
    logging.info(f"Generating {num_frames} frames for prompt: {prompt}")
    output = pipe(
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    )
    return [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in output.frames]

# Save frames as MP4 video
def save_video(frames, out_path, fps):
    logging.info(f"Saving video to {out_path}")
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(out_path, codec="libx264", verbose=False, logger=None)
    return out_path

# Telegram Bot
class TextToVideoBot:
    def __init__(self, token, cfg):
        self.cfg = cfg
        self.pipe = load_pipeline(cfg["model"]["id"], cfg["model"].get("revision"))
        self.bot = Bot(token)
        self.updater = Updater(self.bot, use_context=True)
        dp = self.updater.dispatcher
        dp.add_handler(CommandHandler("start", self.start))
        dp.add_handler(MessageHandler(Filters.text & ~Filters.command, self.on_text))

    def start(self, update: Update, context: CallbackContext):
        keyboard = [["Generate Video", "Help"]]
        markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        update.message.reply_text(
            "Welcome! Click 'Generate Video' then send your prompt, or 'Help' for instructions.",
            reply_markup=markup
        )

    def on_text(self, update: Update, context: CallbackContext):
        text = update.message.text.strip()
        if text.lower() == "help":
            update.message.reply_text(
                "Send any descriptive text and I'll reply with a short (360p) video.",
                reply_markup=ReplyKeyboardRemove()
            )
            return

        update.message.reply_text("Generating video… please wait.", reply_markup=ReplyKeyboardRemove())
        try:
            frames = generate_frames(self.pipe, text, **self.cfg["generation"])
            path = save_video(frames, self.cfg["output"]["path"], self.cfg["output"]["fps"])
            with open(path, "rb") as vid:
                update.message.reply_video(vid)
        except Exception as e:
            logging.error(f"Generation error: {e}")
            update.message.reply_text(f"Video generation failed: {e}")

    def run(self):
        logging.info("Bot polling started")
        self.updater.start_polling()
        self.updater.idle()

if __name__ == "__main__":
    config = load_config("config.yaml")
    TELEGRAM_TOKEN = "5161663037:AAEW27Jyg3aeV2ZCttUET2EIQaDUQIFS9Ds"
    TextToVideoBot(TELEGRAM_TOKEN, config).run()

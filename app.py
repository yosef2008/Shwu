"""
Minimal Text-to-Video Telegram Bot (CPU-only)
Uses diffusers v0.12.1 to avoid accelerate_utils import errors.
"""
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
            # Last attempt
            return func(*args, **kwargs)
        return wrapper
    return deco

# Load configuration
@retry(Exception)
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Load Diffusers pipeline on CPU
@retry(Exception)
def load_pipeline(model_id, revision=None, torch_dtype=torch.float32):
    device = "cpu"
    logging.info(f"Loading model {model_id} on {device}")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        safety_checker=None,
        device_map=device
    )
    pipe.enable_attention_slicing()
    return pipe

# Generate video frames
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

# Save video to file
def save_video(frames, out_path, fps):
    logging.info(f"Saving video to {out_path}")
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(out_path, codec="libx264", verbose=False, logger=None)
    return out_path

# Telegram Bot Class
class TextToVideoBot:
    def __init__(self, token, cfg):
        self.cfg = cfg
        self.pipe = load_pipeline(
            cfg["model"]["id"],
            cfg["model"].get("revision")
        )
        self.bot = Bot(token)
        self.updater = Updater(self.bot, use_context=True)
        dp = self.updater.dispatcher
        dp.add_handler(CommandHandler("start", self.start))
        dp.add_handler(MessageHandler(Filters.text & ~Filters.command, self.on_text))

    def start(self, update: Update, context: CallbackContext):
        keyboard = [["Generate Video", "Help"]]
        markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        update.message.reply_text(
            "Welcome! Click 'Generate Video' then send a prompt, or 'Help' for instructions.",
            reply_markup=markup
        )

    def on_text(self, update: Update, context: CallbackContext):
        text = update.message.text
        if text.lower() == "help":
            update.message.reply_text(
                "Send any text prompt and I will generate a short video (360p).",
                reply_markup=ReplyKeyboardRemove()
            )
            return

        update.message.reply_text(
            "Generating video… Please wait.",
            reply_markup=ReplyKeyboardRemove()
        )
        try:
            frames = generate_frames(
                self.pipe,
                text,
                **self.cfg["generation"]
            )
            path = save_video(
                frames,
                self.cfg["output"]["path"],
                self.cfg["output"]["fps"]
            )
            update.message.reply_video(open(path, "rb"))
        except Exception as e:
            logging.error(f"Generation error: {e}")
            update.message.reply_text(f"Generation failed: {e}")

    def run(self):
        logging.info("Bot polling started")
        self.updater.start_polling()
        self.updater.idle()

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    token = "5161663037:AAEW27Jyg3aeV2ZCttUET2EIQaDUQIFS9Ds"
    TextToVideoBot(token, cfg).run()

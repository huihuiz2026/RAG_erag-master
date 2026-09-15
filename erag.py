from ui import LocalChatbotUI
from logger import Logger


LOG_FILE = "logging.log"
DATA_DIR = "data/data"
AVATAR_IMAGES = ["./assets/user.png", "./assets/bot.png"]


logger = Logger(LOG_FILE)
logger.reset_logs()


ui = LocalChatbotUI(
    logger=logger,
    data_dir=DATA_DIR,
    avatar_images=AVATAR_IMAGES
)

from fastapi import FastAPI
import gradio as gr
app = FastAPI()

io = ui.build()
app = gr.mount_gradio_app(app, io, path="/erag")


#ui.build().launch(
#    share=False,
#    server_name="0.0.0.0",
#    debug=True,
#    show_api=False,
#    favicon_path="./assets/bot.png"
#)

import os
import sys
import time
import gradio as gr
from dataclasses import dataclass
from typing import ClassVar
from ui.theme import JS_LIGHT_THEME, CSS
from logger import Logger
import random
from router import Agent
from utils import extract_tables_and_remainder


@dataclass
class DefaultElement:
    DEFAULT_MESSAGE: ClassVar[dict] = {"text": ""}
    DEFAULT_MODEL: str = ""
    DEFAULT_HISTORY: ClassVar[list] = []
    DEFAULT_DOCUMENT: ClassVar[list] = []

    HELLO_MESSAGE: str = "Hi, 我是你的企业员工助手，请问有什么需要帮助的吗？"
    DEFAULT_STATUS: str = "准备就绪!"
    ANSWERING_STATUS: str = "回答中, 请稍后"
    COMPLETED_STATUS: str = "全部完成"


class LLMResponse:
    def __init__(self) -> None:
        self.router = Agent()

    def _yield_string(self, message: str):
        for i in range(len(message)):
            time.sleep(0.01)
            yield (
                DefaultElement.DEFAULT_MESSAGE,
                [[None, message[: i + 1]]],
                DefaultElement.DEFAULT_STATUS,
            )

    def welcome(self):
        yield from self._yield_string(DefaultElement.HELLO_MESSAGE)

    def empty_message(self):
        yield from self._yield_string(DefaultElement.EMPTY_MESSAGE)

    def stream_response(
        self,
        message: str,
        history
    ):
        answer = []
        rag_response = self.router.text_completion(message)
        if isinstance(rag_response, str):
            yield (
                DefaultElement.DEFAULT_MESSAGE,
                history + [[message, rag_response]],
                DefaultElement.ANSWERING_STATUS,
                "这个问题不属于本系统回答的范围",
                "",)
        else:
            gen_text = ""
            for chunk in rag_response[0]:
                # DashScope / OpenAI 新版 SDK 的 chat.completions 流式返回结构为
                # chunk.choices[0].delta.content，旧代码里的 chunk.choices[0].text
                # 是 Completions（非 chat）接口的字段，在 chat 接口下永远取不到内容，
                # 会导致前端打字机一直空白。同时首尾若干 chunk 的 content 可能为 None，
                # 直接 join 会抛 TypeError，这里做空值保护。
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or ""
                if not text:
                    continue
                answer.append(text)
                gen_text = "".join(answer)

                yield (
                    DefaultElement.DEFAULT_MESSAGE,
                    history + [[message, gen_text]],
                    DefaultElement.ANSWERING_STATUS,
                    "参考信息生成中...",
                    "",)

            tables, context = extract_tables_and_remainder(rag_response[1])
            yield (
                DefaultElement.DEFAULT_MESSAGE,
                history + [[message, gen_text]],
                DefaultElement.COMPLETED_STATUS,
                context,
                "<br>".join(tables),)


class LocalChatbotUI:
    def __init__(
        self,
        logger: Logger,
        data_dir: str = "data/data",
        avatar_images = ["./assets/user.png", "./assets/bot.png"],
    ):
        self._logger = logger
        self._data_dir = os.path.join(os.getcwd(), data_dir)
        self._avatar_images = [
            os.path.join(os.getcwd(), image) for image in avatar_images
        ]
        self._variant = "panel"
        self._llm_response = LLMResponse()

    def _get_respone(
                    self,
                    message,
                    chatbot,
                    progress=gr.Progress(track_tqdm=True),):

        if message["text"] in [None, ""]:
            for m in self._llm_response.welcome():
                yield m
        else:
            sys.stdout = self._logger
            history = chatbot
            for m in self._llm_response.stream_response(message["text"], history):
                yield m

    def _undo_chat(self, history):
        if len(history) > 0:
            history.pop(-1)
            return history
        return DefaultElement.DEFAULT_HISTORY

    def _reset_chat(self):
        gr.Info("Reset chat!")
        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.DEFAULT_STATUS,
            "",
            "",
        )

    def _clear_chat(self):
        gr.Info("Clear chat!", duration=2)
        return (
            DefaultElement.DEFAULT_MESSAGE,
            DefaultElement.DEFAULT_HISTORY,
            DefaultElement.DEFAULT_STATUS,
            "",
            "",
        )

    def _show_hide_setting(self, state):
        state = not state
        label = "Hide Result" if state else "Show Result"
        return (label, gr.update(visible=state), state)

    def _welcome(self):
        for m in self._llm_response.welcome():
            yield m

    def build(self):
        with gr.Blocks(
            theme=gr.themes.Soft(primary_hue="slate"),
            js=JS_LIGHT_THEME,
            css=CSS,
            title="企业员工助手e-rag"
        ) as app:
            gr.HTML("""
                    <div>
                        <h1 style="text-align: center; display:block;">企业员工助手erag</h1>
                    </div>
                    """)            
            with gr.Tab("E-RAG"):
                sidebar_state = gr.State(True)
                with gr.Row(variant=self._variant, equal_height=True):
                    with gr.Column(variant=self._variant, scale=10, visible=sidebar_state.value) as setting:
                        with gr.Row(variant=self._variant, equal_height=False):
                            with gr.Column(variant=self._variant):
                                status = gr.Textbox(
                                    label="运行情况", 
                                    value="准备就绪!", 
                                    interactive=False)
                                


                        with gr.Row(variant=self._variant, equal_height=False):
                            with gr.Column(variant="compact"):
                                contexts = gr.Textbox(
                                    label="参考信息", 
                                    value="", 
                                    interactive=False)
                                table_contexts = gr.HTML("<div>参考表格信息<div>")
                    
                    with gr.Column(scale=30, variant=self._variant):

                        chatbot = gr.Chatbot(
                            layout="bubble",
                            value=[],
                            height=550,
                            scale=2,
                            show_copy_button=True,
                            bubble_full_width=False,
                            avatar_images=self._avatar_images,
                        )

                        with gr.Row(variant=self._variant):
                            message = gr.MultimodalTextbox(
                                value=DefaultElement.DEFAULT_MESSAGE,
                                placeholder="你好，请输入问题",
                                file_types=[".txt", ".pdf", ".csv"],
                                show_label=False,
                                scale=6,
                                lines=1,
                                autofocus=True,
                            )
                        with gr.Row(variant="compact"):
                            ui_btn = gr.Button(
                                value="Hide result"
                                if sidebar_state.value
                                else "Show result",
                                min_width=20,
                            )
                            undo_btn = gr.Button(value="Undo", min_width=20)
                            clear_btn = gr.Button(value="Clear", min_width=20)
                            reset_btn = gr.Button(value="Reset", min_width=20)

            # process event
            clear_btn.click(self._clear_chat, outputs=[message, chatbot, status, contexts, table_contexts])
            undo_btn.click(self._undo_chat, inputs=[chatbot], outputs=[chatbot])
            reset_btn.click(self._reset_chat, outputs=[message, chatbot, status, contexts, table_contexts])

            message.submit(
                self._get_respone,
                inputs=[message, chatbot],
                outputs=[message, chatbot, status,contexts, table_contexts],
            )

            ui_btn.click(
                self._show_hide_setting,
                inputs=[sidebar_state],
                outputs=[ui_btn, setting, sidebar_state],
            )
            app.load(self._welcome, outputs=[message, chatbot, status])

        return app

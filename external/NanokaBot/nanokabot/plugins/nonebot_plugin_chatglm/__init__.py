#coding: utf-8
from nonebot import get_plugin_config
from nonebot.plugin import PluginMetadata

from nonebot import on_command, on_fullmatch, on_regex, require
from nonebot.adapters.onebot.v11 import (
    GROUP,
    GROUP_ADMIN,
    GROUP_OWNER,
    GroupMessageEvent,
    Message,
    MessageSegment,
)
from nonebot.log import logger
from nonebot.matcher import Matcher
from nonebot.params import CommandArg, Depends, RegexStr
from nonebot.permission import SUPERUSER

from .config import Config
from .chatglm import chatglm_model, request_zhipuai_chatglm

__llm_version__ = "v0.1.0"
__llm_usages__ = f"""
[智能问答] 向 ChatGLM-4-Flash 模型提问
[查看模型] 查看当前使用模型""".strip()
__llm_model__ = "glm-4-flash"

__plugin_meta__ = PluginMetadata(
    name="nonebot_plugin_chatglm",
    description="使用 ChatGLM-4-Flash 模型进行对话",
    usage=__llm_usages__,
    config=Config,
)

config = get_plugin_config(Config)

request_llm = on_regex(r"^/智能问答 (.*)$", permission=GROUP, priority=5, block=True)
view_model = on_command("查看模型", aliases={"模型", "当前模型"}, permission=GROUP, priority=7)

@request_llm.handle()
async def handle_request_llm(
    event: GroupMessageEvent,
    matcher: Matcher,
    content: str = RegexStr(),
):
    """
    处理智能问答请求
    """
    content = content.strip()

    if not content:
        await matcher.finish("内容不能为空")

    success, response = request_zhipuai_chatglm(content)
    if not success:
        await matcher.finish(response)

    await matcher.finish(MessageSegment.text(response))

@view_model.handle()
async def handle_view_model():
    """
    查看当前使用的模型
    """
    await view_model.finish(f"当前使用的模型是：{__llm_model__}")
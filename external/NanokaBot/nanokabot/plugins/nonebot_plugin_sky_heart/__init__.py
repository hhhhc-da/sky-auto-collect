from typing import Annotated

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
from .crawler import WebCrawler

__plugin_meta__ = PluginMetadata(
    name="nonebot_plugin_sky_heart",
    description="",
    usage="",
    config=Config,
)

config = get_plugin_config(Config)
webcrawler = WebCrawler()

__help_version__ = "v0.1.0"
__help_usages__ = f"""
[帮助] 查看 NanokaBot 的所有指令内容""".strip()

rquest_heart = on_command("取心", aliases={"取爱心"}, priority=7, block=True)
set_url = on_regex(r"^/设置网址 (.*)$", permission=GROUP, priority=5, block=True)
check_url = on_fullmatch("/查看网址", permission=GROUP, priority=5, block=True)
del_url = on_regex(r"^/删除网址 (.*)$", permission=GROUP, priority=5, block=True)
clear_url = on_fullmatch("/清空网址", permission=GROUP, priority=5, block=True)

@rquest_heart.handle()
async def handle_request_heart(event: GroupMessageEvent, args: Annotated[Message, CommandArg()]):
    """
    处理取心请求
    """
    status, code = webcrawler.crawl_main(url="http://vip.gyzhax.cn/gy?key=SqIsr71MpxZL", headless=False)

    # 这里可以添加更多的处理逻辑
    await rquest_heart.finish("请在五分钟内完成操作\n\n状态: {}\n好友码: {}".format(status, code))

@set_url.handle()
async def handle_set_url(
    event: GroupMessageEvent,
    matcher: Matcher,
    content: str = RegexStr(),
):
    """
    处理设置 URL 请求
    """
    content = content.strip()

    if not content:
        await set_url.finish("内容不能为空")

    # 这里可以添加更多的处理逻辑
    await set_url.finish(MessageSegment.text(f"设置的 URL 为: {content}"))

@check_url.handle()
async def handle_check_url(event: GroupMessageEvent):
    """
    处理查看 URL 请求
    """
    # 这里可以添加更多的处理逻辑
    await check_url.finish("当前没有设置任何 URL")

@del_url.handle()
async def handle_del_url(
    event: GroupMessageEvent,
    matcher: Matcher,
    content: str = RegexStr(),
):
    """
    处理删除 URL 请求
    """
    content = content.strip()

    if not content:
        await del_url.finish("内容不能为空")

    # 这里可以添加更多的处理逻辑
    await del_url.finish(MessageSegment.text(f"已删除 URL: {content}"))

@clear_url.handle()
async def handle_clear_url(event: GroupMessageEvent):
    """
    处理清空 URL 请求
    """
    # 这里可以添加更多的处理逻辑
    await clear_url.finish("已清空所有 URL")
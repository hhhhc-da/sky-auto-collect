#coding: utf-8
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

__help_version__ = "v0.1.0"
__help_usages__ = f"""
[帮助] 查看 NanokaBot 的所有指令内容""".strip()

__plugin_meta__ = PluginMetadata(
    name="nonebot_plugin_nanokabot_help",
    description="用来返回可以的帮助信息",
    usage=__help_usages__,
    config=Config,
)

config = get_plugin_config(Config)

help_handle = on_command("帮助", aliases={"help"}, permission=GROUP, priority=7)

@help_handle.handle()
async def handle_help(event: GroupMessageEvent, args: Annotated[Message, CommandArg()]):
    """
    处理帮助请求
    """
    help_text = '''欢迎使用 NanokaBot！

帮助类指令：
- (MISC) 帮助/help: 查看所有指令内容

人工智能聊天类指令：
- (CHAT) 智能问答: 向 ChatGLM 提问
- (CHAT) 查看模型: 查看当前使用模型
Web Site: https://open.bigmodel.cn/

今日运势指令：
- (FRTN) 今日运势/抽签/运势: 一般抽签
- (FRTN) xx抽签: 指定主题抽签
- (FRTN) 设置xx签: 设置群抽签主题
- (FRTN) 重置主题: 重置群抽签主题
- (FRTN) 主题列表: 查看可选的抽签主题
- (FRTN) 查看主题: 查看群抽签主题
GitHub Infomation:
Author-: MinatoAquaCrews
Project: nonebot_plugin_fortune

光遇辅助指令：
- (SKY-) 取心: 取一颗爱心
- (SKY-) 设置网址: 设置 Sky API Key
- (SKY-) 查看网址列表: 查看当前 Sky API Key
- (SKY-) 删除网址: 删除指定的 Sky API Key
- (SKY-) 清空网址: 清空所有的 Sky API Key
'''

    await help_handle.finish(help_text)
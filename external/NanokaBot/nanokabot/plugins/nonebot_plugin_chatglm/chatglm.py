from zhipuai import ZhipuAI

# 这里需要你自己去弄这个API Key，好消息是它是免费的
chatglm_model = ZhipuAI(api_key="")

def request_zhipuai_chatglm(content='') -> tuple[bool, str]:
    """
    请求 ZhipuAI 的 ChatGLM-4-Flash 模型进行对话
    """
    if len(content) == 0:
        return False, "内容不能为空"

    response = chatglm_model.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
            {"role": "user", "content": content}
        ],
    )

    return True, response.choices[0].message.content
import os
from llms.siliconflow_wrapper import sfa_llm
from openai import OpenAI


base_url = os.environ.get('LLM_API_BASE', "")
token = os.environ.get('LLM_API_KEY', "")

if token:
    client = OpenAI(api_key=token, base_url=base_url)
else:
    client = OpenAI(base_url=base_url)

def wraper_llm(messages: list, model: str, logger=None, **kwargs) -> str:
    base_url = os.environ.get('LLM_API_BASE', "")
    if 'siliconflow' in base_url:
        return sfa_llm(messages=messages,model=model,logger=logger,kwargs=kwargs)
    else:
        return openai_llm(messages=messages,model=model,logger=logger,kwargs=kwargs)

def openai_llm(messages: list, model: str, logger=None, **kwargs) -> str:

    if logger:
        logger.debug(f'messages:\n {messages}')
        logger.debug(f'model: {model}')
        logger.debug(f'kwargs:\n {kwargs}')

    try:
        response = client.chat.completions.create(messages=messages, model=model, **kwargs)

    except Exception as e:
        if logger:
            logger.error(f'openai_llm error: {e}')
        return ''

    if logger:
        logger.debug(f'result:\n {response.choices[0]}')
        logger.debug(f'usage:\n {response.usage}')

    return response.choices[0].message.content

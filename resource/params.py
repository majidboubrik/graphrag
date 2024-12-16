import dataiku

def do(payload, config, plugin_config, inputs):
    client = dataiku.api_client()
    project = client.get_default_project()
    chat_llms = project.list_llms(purpose='GENERIC_COMPLETION')
    emnedding_llms = project.list_llms(purpose='TEXT_EMBEDDING_EXTRACTION')

    if payload.get('parameterName') == 'chat_completion_llm':
        choices = [
            {"value": llm["id"], "label": llm["friendlyName"]}
            for llm in chat_llms
        ]
        return {"choices": choices}

    elif payload.get('parameterName') == 'embedding_llm':
        choices = [
            {"value": llm["id"], "label": llm["friendlyName"]}
            for llm in emnedding_llms
        ]
        return {"choices": choices}

    return {"choices": []}
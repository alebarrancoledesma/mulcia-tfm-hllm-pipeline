import requests
import os
from dotenv import load_dotenv
from openai import OpenAI
from hllm_pipeline.utils import read_text_file
import concurrent.futures
import time
import tiktoken


load_dotenv()

############################
# PROMPT GENERATION
############################

def prompt_gen(template_path, vars):
    
    template = read_text_file(template_path)

    for v in vars.items():
        template = template.replace("{{ " + v[0] + " }}", v[1])
    
    return template


############################
# LLM CALL
############################

def llm_generate(provider, model, prompt, temperature=None, response_format=None):

    match provider:

        case "openai":
            return openai_generate(prompt, model=model, temperature=temperature, response_format=response_format)

        case _:
            raise NotImplementedError(f"Provider '{provider}' is not supported.")


def llm_generate_parallel(provider, model, prompt_list, temperature=None, response_format=None, requests_per_minute=30):

    result = []

    with concurrent.futures.ThreadPoolExecutor() as executor:

        futures = []
        
        for n, prompt in enumerate(prompt_list):
            # Llamar a función síncrona de petición única
            futures.append(executor.submit(llm_generate, provider, model, prompt, temperature, response_format))

            idx = n+1
            print(f"Enviando prompt {idx}")

            # Limitar rate per minute
            if requests_per_minute:
                time.sleep(60.0 / requests_per_minute)

        result = [future.result() for future in futures]

    return result


############################
# EACH CREATOR FUNCTIONS
############################


def ollama_generate(prompt, model="qwen2.5:7b", url = "http://localhost:11434/api/generate"):
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False 
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["response"]

def openai_generate(prompt, model="gpt-5.6-sol", temperature=None, response_schema=None, api_key=None, reasoning_effort="medium"):
    try:
        load_dotenv()

        api_key = api_key or os.getenv("OPENAI_KEY")
        client = OpenAI(api_key=api_key)

        params = {
            "model": model,
            "input": prompt,
            "reasoning": {
                "effort": reasoning_effort
            },
        }

        if temperature is not None:
            params["temperature"] = temperature

        if response_schema is not None:
            params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "response",
                    "schema": response_schema,
                    "strict": True
                }
            }

        response = client.responses.create(**params)

        return response.output_text

    except Exception as e:
        print(f"Error OpenAI: {e}")
        return '{"content": [{"status": "ERROR"}]}'



def openai_generate_parallel(prompt_list, model="gpt-5", api_key=os.getenv("OPENAI_KEY"), temperature=None, response_format=None, requests_per_minute=30):

    result = []

    with concurrent.futures.ThreadPoolExecutor() as executor:

        futures = []
        
        for n, prompt in enumerate(prompt_list):
            # Llamar a función síncrona de petición única
            futures.append(executor.submit(openai_generate, prompt, model, api_key,temperature, response_format))

            idx = n+1
            print(f"Enviando prompt {idx}")

            # Limitar rate per minute
            time.sleep(60.0 / requests_per_minute)


        result = [future.result() for future in futures]

    return result


############################
# PRICING
############################

def get_estimated_prices(model, paid, prompt_input, prompt_output):

    input_tokens, output_tokens = get_estimated_token_number(model, prompt_input, prompt_output)
    
    if not paid:
        return (input_tokens, 0.), (output_tokens, 0.)
    
    else:
        token_prices = get_llm_model_price(model)
        input_price = input_tokens * token_prices['price_1m_input_tokens'] / 1000000
        output_price = output_tokens * token_prices['price_1m_output_tokens'] / 1000000

        return (input_tokens, input_price), (output_tokens, output_price)


def get_estimated_token_number(model, prompt_input, prompt_output):

    try:
        return len(tiktoken.encoding_for_model(model).encode(str(prompt_input))), len(tiktoken.encoding_for_model(model).encode(str(prompt_output)))
    except:
        return len(tiktoken.get_encoding("cl100k_base").encode(str(prompt_input))), len(tiktoken.get_encoding("cl100k_base").encode(str(prompt_output)))


def get_llm_model_price_map():

    load_dotenv()
    response = requests.get('https://artificialanalysis.ai/api/v2/data/llms/models', headers = {'x-api-key': os.getenv("ARTIFICIAL_ANALYSIS_KEY")})

    if response.status_code == 200:
        return {model['slug']: model['pricing'] for model in response.json()['data']}
    else:
        return {}

def get_llm_model_price(slug):
    return get_llm_model_price_map().get(slug, None)

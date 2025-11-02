import tiktoken
def count_tokens(text: str,deployment_name: str ="cl100k_base") -> int:
    """counts number of tokens that will be spent when sent to the openai model

    Args:
        text (str): text to be tokenized
        deployment_name (str, optional): name of the token encoding. Defaults to "cl100k_base".

    Raises:
        TypeError: if text is not a string

    Returns:
        int: number of tokens the text will be tokenized to
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected a string, but got {type(text).__name__}")
    
    encoding = tiktoken.get_encoding(deployment_name) 
    return len(encoding.encode(text))
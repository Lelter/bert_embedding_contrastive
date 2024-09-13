#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('LLM-Research/gemma-2-9b-it',cache_dir='/data/llm/')
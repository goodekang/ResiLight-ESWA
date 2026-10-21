import json
import time

from .schema import GBNF, JSON_SCHEMA


class Backend:
    def generate(self, messages, seed=None):
        raise NotImplementedError


class LlamaCppBackend(Backend):
    def __init__(self, model_path, temperature=0.2, n_ctx=4096, n_gpu_layers=-1, max_tokens=256):
        from llama_cpp import Llama, LlamaGrammar
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
        self.grammar = LlamaGrammar.from_string(GBNF)
        self.temperature, self.max_tokens = temperature, max_tokens

    def generate(self, messages, seed=None):
        out = self.llm.create_chat_completion(
            messages=messages, temperature=self.temperature, max_tokens=self.max_tokens,
            grammar=self.grammar, seed=seed)
        return out["choices"][0]["message"]["content"]


class OpenAIBackend(Backend):
    def __init__(self, model="gpt-4o", temperature=0.2, max_tokens=256):
        from openai import OpenAI
        self.client = OpenAI()
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens

    def generate(self, messages, seed=None):
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens, seed=seed,
            response_format={"type": "json_schema", "json_schema": {"name": "decision", "schema": JSON_SCHEMA, "strict": False}})
        return r.choices[0].message.content


class HFBackend(Backend):
    def __init__(self, model_id, temperature=0.2, max_tokens=256, load_in_4bit=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.tok = AutoTokenizer.from_pretrained(model_id)
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16) if load_in_4bit else None
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=q, device_map="auto")
        self.temperature, self.max_tokens = temperature, max_tokens

    def generate(self, messages, seed=None):
        import torch
        if seed is not None:
            torch.manual_seed(seed)
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(ids, do_sample=True, temperature=self.temperature, max_new_tokens=self.max_tokens)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def make_backend(cfg_sup, model_path=None):
    kind = cfg_sup["backend"]
    if kind == "llama_cpp":
        return LlamaCppBackend(model_path, cfg_sup["temperature"])
    if kind == "openai":
        return OpenAIBackend(cfg_sup["model"], cfg_sup["temperature"])
    if kind == "hf":
        return HFBackend(cfg_sup["model"], cfg_sup["temperature"])
    raise ValueError(kind)


def timed(backend, messages, budget_s, seed=None):
    t0 = time.perf_counter()
    try:
        text = backend.generate(messages, seed)
    except Exception:
        return None, time.perf_counter() - t0
    dt = time.perf_counter() - t0
    return (text if dt <= budget_s else None), dt

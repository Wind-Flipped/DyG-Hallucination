
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import torch
from torch.cuda.amp import autocast

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.environ['VLLM_USE_MODELSCOPE'] = 'True'


class Qwen:
    history: list = []

    def __init__(self, max_tokens=1024, temperature=0, top_p=0.95,
                 model='', max_model_len=2048, **kwargs):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model = model
        self.max_model_len = max_model_len
        self.llm = LLM(model=self.model, tokenizer=None, max_model_len=self.max_model_len, trust_remote_code=False)
        print("Successfully get remote api for Qwen LLM ")

    def get_completion(self, prompts):
        stop_token_ids = [151329, 151336, 151338]

        sampling_params = SamplingParams(temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens,
                                         stop_token_ids=stop_token_ids)
        outputs = self.llm.generate(prompts, sampling_params)

        return outputs

    def __call__(self, questions):
        results = []
        if questions:
            responses = self.get_completion(questions)
            for response in responses:
                result = response.outputs[0].text
                results.append(result)
            print(results)
            return results
        else:
            print("Some error occur in apis")
            return None

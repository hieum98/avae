# Licensed under the MIT license.

from typing import Dict, List, Union
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer
import numpy as np
import math
import json


def load_vLLM_model(model_ckpt, seed, tensor_parallel_size=1, half_precision=False, max_num_seqs=256):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    if half_precision:
        llm = LLM(
            model=model_ckpt,
            dtype="half",
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=16,
            max_model_len=16384,
        )
    else:
        llm = LLM(
            model=model_ckpt,
            tensor_parallel_size=tensor_parallel_size,
            seed=seed,
            trust_remote_code=True,
            max_num_seqs=max_num_seqs,
            swap_space=16,
            max_model_len=16384,
        )

    return tokenizer, llm


def generate_with_vLLM_model(
    model,
    input: Union[str, List[str], List[Dict[str, str]], List[List[Dict[str, str]]]],
    guided_decoding_params=None,
    temperature=0.8,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1,
    n=1,
    max_tokens=256,
    logprobs=1,
    stop=[],
):
    sampling_params = SamplingParams(
        temperature=temperature,
        guided_decoding=guided_decoding_params,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        n=n,
        logprobs=logprobs,
        max_tokens=max_tokens,
        stop=stop,
    )
    if isinstance(input, str):
        output = model.generate(input, sampling_params, use_tqdm=False)
    elif isinstance(input, list):
        if all(isinstance(i, str) for i in input):
            output = model.generate(input, sampling_params, use_tqdm=False)
        elif all(isinstance(i, dict) for i in input):
            output = model.chat(input, sampling_params, use_tqdm=False)
        elif all(isinstance(i, list) for i in input):
            assert all(isinstance(j, dict) for i in input for j in i), "All elements in the nested list must be dictionaries."
            output = model.chat(input, sampling_params, use_tqdm=False)
        else:
            raise ValueError("Input must be a string, list of strings, list of dictionaries, or list of lists of dictionaries.")
    return output


class BlackBoxIOSystem:
    def __init__(
        self,
        model_ckpt,
        seed=42,
        tensor_parallel_size=1,
        half_precision=True,
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        max_num_seqs=256,
    ):
        self.tokenizer, self.model = load_vLLM_model(
            model_ckpt, seed, tensor_parallel_size, half_precision, max_num_seqs
        )
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        self.call_counter = 0
        self.token_counter = 0
    
    def generate(self, model_input, stop_tokens, num_return: int, max_tokens: int=1024, guided_decoding_params=None):
        if isinstance(model_input, str):
            vllm_response = generate_with_vLLM_model(
                    self.model,
                    input=model_input,
                    temperature=self.temperature,
                    guided_decoding_params=guided_decoding_params,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    n=num_return,
                    max_tokens=max_tokens,
                    stop=stop_tokens,
                )
            io_output_list = [o.text for o in vllm_response[0].outputs]
            self.call_counter += 1
            self.token_counter += sum([len(o.token_ids) for o in vllm_response[0].outputs])
        elif isinstance(model_input, list):
            vllm_response = generate_with_vLLM_model(
                self.model,
                input=model_input,
                guided_decoding_params=guided_decoding_params,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                n=num_return,
                max_tokens=max_tokens,
                stop=stop_tokens,
            )
            io_output_list = [
                [o.text for o in resp_to_single_input.outputs] for resp_to_single_input in vllm_response
            ]
            self.call_counter += 1
            self.token_counter += sum(
                [
                    sum([len(o.token_ids) for o in resp_to_single_input.outputs])
                    for resp_to_single_input in vllm_response
                ]
            )
        return io_output_list
    
        

if __name__ == "__main__":
    import time
    import os
    from src.data_modules.templates import GENERATE_STYLE_COUNTERFACTUAL_PROMPT, STYLE_COUNTERFACTUAL_IN_CONTEXT_EXAMPLES, StyleTransferReply

    model_ckpt = "microsoft/phi-4"
    seed = 42
    tensor_parallel_size = 1 # With Guided Decoding, the tensor_parallel_size should be 1 (this will be fixed in the future)
    half_precision = False
    temperature = 0.8
    top_p = 0.95
    top_k = 40
    max_num_seqs = 256

    model = BlackBoxIOSystem(
        model_ckpt=model_ckpt,
        seed=seed,
        tensor_parallel_size=tensor_parallel_size,
        half_precision=half_precision,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_num_seqs=max_num_seqs,
    )

    input_text = "Hey, wanna grab coffee later? I found this cool new spot near the park."
    messages = [
        {"role": "system", "content": GENERATE_STYLE_COUNTERFACTUAL_PROMPT,},
    ]
    # for example in in_context_examples:
    #     messages.append({"role": "user", "content": example["input_text"]})
    #     output = Reply(rewrited_text=example["rewrited_text"], style_comparison=example["style_comparison"])
    #     output = output.model_dump_json()
    #     messages.append({"role": "assistant", "content": output})
    messages.append({"role": "user", "content": input_text})

    messages = [messages]*10

    stop_tokens = []
    num_return = 1
    max_tokens = 1024

    json_schema = StyleTransferReply.model_json_schema()
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    output = model.generate(
        model_input=messages,
        guided_decoding_params=guided_decoding_params,
        stop_tokens=stop_tokens,
        num_return=num_return,
        max_tokens=max_tokens,
    )

    instance = StyleTransferReply.model_validate_json(output[0][0])
    breakpoint()

    

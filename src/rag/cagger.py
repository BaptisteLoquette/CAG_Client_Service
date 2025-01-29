import os
import re
import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer

class CAG:
    """
    Here is the class for a Cache Augmented Generation approach
        - The advantage of this approach is that it does not requires pre-indexing of Documents
        - The model is a Qwen2.5-0.5B-Instruct model with 32k context window, enough to handle long context
    """
    def __init__(self, text_content:list[str], model_name:str="Qwen/Qwen2.5-0.5B-Instruct") -> None:
        self.model_name = model_name
        self.text_content = text_content
        self.knowledge = "\n---------------------\n".join(self.text_content)
        self.model, self.tokenizer = self.get_model_and_tokenizer()
        self.prompt = self.get_prompt()
    
    def perform_cag(self, question = "How can I subscribe and update my payment details?"):
        cache = self.get_kv_cache(self.tokenizer, self.prompt)
        origin_len = cache.key_cache[0].shape[-2]
        self.clean_up(cache, origin_len)
        origin_len = cache.key_cache[0].shape[-2]
        input_ids_q = self.tokenizer(question + "\n", return_tensors="pt").input_ids.to("cuda")
        gen_ids_q = self.generate(input_ids_q, cache, max_new_tokens=512)
        answer = self.tokenizer.decode(gen_ids_q[0], skip_special_tokens=False)
        answer = self.extract_text_between_markers(answer)

        return answer
    
    def get_prompt(self):


        prompt = f"""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are an assistant that gives concise answers and help non-technicalusers based on given context. In the context of a Client Service Assistance, you must be concise and precise.
        You must follow the given instructions.
        You must only base your answers on the given knowledge.
        If there is not enough given context to answer accurately the user's query given the knowledge, you must output : "out of scope".
        If the query is not relevant given the knowledge, you must output : "out of scope".
        If the query is not related or outside of the given knowledge, you must output : "out of scope".
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Question:
        <|eot_id|>
        """.strip()

        return prompt
    
    def get_model_and_tokenizer(self):
        """
        Load the model and the tokenizer. Default model is Qwen2.5-0.5B-Instruct.

        Args:
            model_name:str -> The name of the model to load

        Returns:
            model:AutoModelForCausalLM -> The loaded model
            tokenizer:AutoTokenizer -> The loaded tokenizer
        """
        model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
                temperature=0.0,
                do_sample=True
            )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return model, tokenizer
    
    def generate(self, input_ids: torch.Tensor, past_key_values:DynamicCache, max_new_tokens: int = 50) -> torch.Tensor:
        device = self.model.model.embed_tokens.weight.device
        origin_len = input_ids.shape[-1]
        input_ids = input_ids.to(device)
        output_ids = input_ids.clone()
        next_token = input_ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                logits = out.logits[:, -1, :]
                token = torch.argmax(logits, dim=-1, keepdim=True)
                output_ids = torch.cat([output_ids, token], dim=-1)
                past_key_values = out.past_key_values
                next_token = token.to(device)

                if self.model.config.eos_token_id is not None and token.item() == self.model.config.eos_token_id:
                    break
        return output_ids[:, origin_len:]
    
    def get_kv_cache(self, tokenizer:AutoTokenizer, prompt: str) -> DynamicCache:
        device = self.model.model.embed_tokens.weight.device
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        cache = DynamicCache()

        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True
            )
        return cache

    def clean_up(self, cache: DynamicCache, origin_len: int):
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][:, :, :origin_len, :]
            cache.value_cache[i] = cache.value_cache[i][:, :, :origin_len, :]

    def extract_text_between_markers(self, text):
        try:
            match1 = re.search(r"<\|start_header_id\|>Answer:(.*?)<\|eot_id\|>", text, re.DOTALL)
            if match1 and len(match1.group(1).strip()) > 40:
                return (match1.group(1).strip(), None)
            else:
                return [None, text]
        except Exception as e:
            # Handle the exception or log it
            return [None, f"Error occurred: {str(e)}"]

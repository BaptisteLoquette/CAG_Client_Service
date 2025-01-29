from transformers import AutoModelForCausalLM, AutoTokenizer

class CAG:
    """
    Here is the class for a Cache Augmented Generation approach
        - The advantage of this approach is that it does not requires pre-indexing of Documents
        - The model is a Qwen2.5-0.5B-Instruct model with 32k context window, enough to handle long context
    """
    def __init__(self, text_content, model_name="Qwen/Qwen2.5-0.5B-Instruct", 
                question = "How can I subscribe and update my payment details?",
                answer_instruction = """You are an helpfull assistant that gives give answers and help the user based on given context. You must ouput in French. Only output the answer, no other text.
                If the user's query is not a real retrieval query or not related or outside (not any information in the given context) of the given contexts below, you must output "out of scope"."""
                ) -> None:
        self.model_name = model_name
        self.text_content = text_content
        self.question = question
        self.answer_instruction = answer_instruction
        self.model, self.tokenizer = self.get_model_and_tokenizer(model_name)
        self.model.eval()
        self.prompt = self.get_prompt()
        self.messages = [
            {"role": """system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant, designed for non technical people.
             You are an assistant, your task is to give answers and help the user based on the given context.
             Your task is to output the accurate answer that respond to the user question.
             If the user's query is not related or outside of the given contexts, you must output "out of scope".
             You must not ouput your reasoning, or any other unneeded text.
             Don't output false or invented information.
             """},
            {"role": "user", "content": self.prompt}
        ]

    def get_prompt(self):
        knowledge = "\n---------------------\n".join(self.text_content)


        prompt = f"""

        INSTRUCTIONS:
        {self.answer_instruction}

        Context information is below.
        Context:
        ------------------------------------------------
        {knowledge}
        ------------------------------------------------

        Question:
        {self.question}

        Factual answer or "out of scope" :
        """

        return prompt

    def get_model_and_tokenizer(self, model_name):
        """
        Load the model and the tokenizer. Default model is Qwen2.5-0.5B-Instruct.

        Args:
            model_name:str -> The name of the model to load

        Returns:
            model:AutoModelForCausalLM -> The loaded model
            tokenizer:AutoTokenizer -> The loaded tokenizer
        """
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                temperature=0.0
            )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return model, tokenizer

    def get_answer(self):
        text = self.tokenizer.apply_chat_template( 
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response
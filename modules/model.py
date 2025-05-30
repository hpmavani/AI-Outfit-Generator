from typing import List
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch

input_file = "submission\\fashion-model\\checkpoints\\epochv2_3.pt"

class ModelHandler: 
    def __init__(self): 
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        checkpoint = torch.load(input_file, map_location=torch.device('cpu'))
    
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[SEP]', '[OUTFIT_END]']})
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval() 
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def predict(self, input: List[str]) -> str:
        input_string, tokenized_input = self.tokenize_input(input)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=tokenized_input["input_ids"],
                attention_mask=tokenized_input["attention_mask"],
                max_length=100,      # Change depending on expected description length
                num_return_sequences=5,
                do_sample=True,     # Enables sampling (creative generation)
                top_k=50,           # Limits sampling pool
                top_p=0.95,         # Nucleus sampling
                temperature=0.9,    # Controls randomness
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.2
            )
        generated_texts = [self.tokenizer.decode(g, skip_special_tokens=False) for g in output_ids]
        cleaned_text = []
        for g in generated_texts: 
            end_index = g.find('[OUTFIT_END]')
            if end_index != -1:
                text = g[len(input_string):end_index]
                if text != "":
                    cleaned_text.append(text)
                
        print(cleaned_text)
        
        if len(cleaned_text) != 0:
            return cleaned_text[0]
        else: 
            return ""
    
    def tokenize_input(self, input: List[str]): 
        input_string = self.create_input_string(input)
        return input_string, self.tokenizer(input_string, return_tensors='pt', padding=True, truncation=True)
        
    def create_input_string(self, items: List[str]): 
        input_string = ""
        for i in items: 
            input_string += f"{i}[SEP]"
        print(input_string)
        return input_string
         
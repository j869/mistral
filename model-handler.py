#!/usr/bin/env python3
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time

class MistralHandler:
    def __init__(self):
        print("🤖 Loading Mistral 7B model...", flush=True)
        self.model_name = "mistralai/Mistral-7B-v0.1"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
            )
            
            print("✅ Mistral 7B loaded successfully!", flush=True)
            
        except Exception as e:
            print(f"❌ Error loading model: {e}", flush=True)
            sys.exit(1)
    
    def generate(self, prompt, max_tokens=150, temperature=0.7):
        try:
            start_time = time.time()
            
            # Create prompt template
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            outputs = self.pipe(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            response_time = time.time() - start_time
            
            full_response = outputs[0]['generated_text']
            response = full_response.split("[/INST]")[-1].strip()
            
            return {
                "success": True,
                "response": response,
                "response_time": f"{response_time:.2f}s",
                "tokens_generated": len(self.tokenizer.encode(response)),
                "full_prompt": prompt
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": "0s"
            }

def main():
    handler = MistralHandler()
    print("✅ Ready to accept requests...", flush=True)
    
    try:
        while True:
            # Read from stdin
            line = sys.stdin.readline().strip()
            if not line:
                continue
                
            try:
                request = json.loads(line)
                prompt = request.get('prompt', '')
                max_tokens = request.get('max_tokens', 150)
                temperature = request.get('temperature', 0.7)
                
                result = handler.generate(prompt, max_tokens, temperature)
                print(json.dumps(result), flush=True)
                
            except json.JSONDecodeError:
                error_result = {
                    "success": False,
                    "error": "Invalid JSON request"
                }
                print(json.dumps(error_result), flush=True)
                
    except KeyboardInterrupt:
        print("👋 Shutting down...", flush=True)
    except Exception as e:
        print(f"❌ Fatal error: {e}", flush=True)

if __name__ == "__main__":
    main()
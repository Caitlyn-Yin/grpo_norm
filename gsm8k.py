# gsm8k.py
import re
import random
import json
from typing import Dict, Any, List, Optional
from datasets import load_dataset

FINAL_RE = re.compile(r"####\s*([\-]?\d+(?:,\d{3})*(?:\.\d+)?)")

def _extract_final_answer(sol: str) -> Optional[str]:
    """Extract final numerical answer from GSM8K answer"""
    m = FINAL_RE.search(sol or "")
    if not m: 
        return None
    s = m.group(1).replace(",", "")
    return s  

class GSM8K:
    def __init__(self, split: str = "train", include_answer: bool = False, 
                 include_reasoning: bool = True, few_shot: bool = True, 
                 num_shots: int = 2, seed: Optional[int] = None, 
                 cot: bool = True, template: str = "qa"):
        self.split = split
        self.include_answer = include_answer
        self.include_reasoning = include_reasoning
        self.few_shot = few_shot
        self.num_shots = num_shots
        self.seed = seed
        self.cot = cot
        self.template = template
        
        if seed is not None:
            random.seed(seed)

        if self.few_shot:
            self.few_shot_prompt = self._build_few_shot_prompt()
        else:
            self.few_shot_prompt = ""
        
        self.dataset = self._load_and_process_dataset()
    
    def _load_and_process_dataset(self):
        dataset = load_dataset("gsm8k", "main", split=self.split)
        
        def process_example(example, idx):
            question = example['question']
            answer = example['answer']
            
            answer_delim = "#### "
            if answer_delim in answer:
                reasoning = answer.split(answer_delim)[0].strip()
                final_answer = answer.split(answer_delim)[-1].strip()
            else:
                reasoning = answer.strip()
                final_answer = _extract_final_answer(answer) or ""
            
            if self.template == 'qa':
                prompt = self._format_qa_example(question, 
                                                reasoning if self.include_reasoning else None,
                                                final_answer if self.include_answer else None)
            elif self.template == 'code':
                prompt = self._format_code_example(question,
                                                  reasoning if self.include_reasoning else None,
                                                  final_answer if self.include_answer else None)
            else:
                prompt = f"Question: {question}\nAnswer:"
            
            if self.few_shot:
                prompt = self.few_shot_prompt + prompt
            
            return {
                'prompt': prompt,
                'final_answer': final_answer,  
                'question': question,
                'raw_answer': answer,
            }
        
        dataset = dataset.map(process_example, 
                            with_indices=True, 
                            load_from_cache_file=False)
        
        return dataset
    
    def _format_qa_example(self, question: str, reasoning: Optional[str] = None, 
                          answer: Optional[str] = None) -> str:
        
        example = f"Question: {question}\nSolution: "
        
        if self.cot:
            example += "Let's think step by step. "
        
        if reasoning is not None:
            reasoning = self._clean_reasoning(reasoning)
            example += f"{reasoning} "
        
        if answer is not None:
            example += f"#### The final answer is {answer}\n\n"
        
        return example
    
    def _format_code_example(self, question: str, reasoning: Optional[str] = None,
                           answer: Optional[str] = None) -> str:
        example = f'Question: {question}\n\n# solution in Python:\n\ndef solution():\n    """{question}"""\n'
        
        if reasoning is not None:
            example += f'    # {reasoning}\n'
        
        if answer is not None:
            example += f'    return {answer}\n\n'
        
        return example
    
    def _clean_reasoning(self, text: str) -> str:
        text = re.sub(r'<<.*?>>', '', text)
        text = re.sub(r'<<.*?>>', '', text)
        text = '. '.join(text.split('\n'))
        text = re.sub(r'\.+', '.', text)
        text = text.strip()
        return text
    
    def _build_few_shot_prompt(self) -> str:
        examples = self._get_few_shot_examples()
        
        selected = random.sample(examples, min(self.num_shots, len(examples)))
        
        prompt = ""
        for ex in selected:
            if self.template == 'qa':
                prompt += self._format_qa_example(ex['question'], 
                                                 ex['reasoning'], 
                                                 ex['answer'])
            elif self.template == 'code':
                prompt += self._format_code_example(ex['question'],
                                                   ex['reasoning'],
                                                   ex['answer'])
        
        return prompt
    
    def _get_few_shot_examples(self) -> List[Dict[str, str]]:
        examples = [
            {
                'question': "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                'reasoning': "There are 15 trees originally. After planting, there will be 21 trees. So the workers planted 21 - 15 = 6 trees",
                'answer': "6"
            },
            {
                'question': "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                'reasoning': "There are originally 3 cars. Then 2 more cars arrive. So there are 3 + 2 = 5 cars in total",
                'answer': "5"
            },
        ]
        
        return examples
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[idx]
    
    def shuffle(self, seed: Optional[int] = None):
        if seed is not None:
            self.dataset = self.dataset.shuffle(seed=seed)
        else:
            self.dataset = self.dataset.shuffle()
        return self

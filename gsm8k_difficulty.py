# download the preprocessed dataset (judge the difficulty by 7B model)
import os
import re
import random
import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Optional, Dict, Any, List

class GSM8KDifficulty:
    """
    Load difficulty-graded GSM8K dataset
    """
    
    def __init__(self, 
                 difficulty: str = "easy",  # easy, medium, hard
                 data_dir: str = "data/gsm8k_difficulty_subsets",
                 split: str = "train",
                 include_answer: bool = False,
                 include_reasoning: bool = True,
                 few_shot: bool = True,
                 num_shots: int = 2,
                 seed: Optional[int] = 42,
                 cot: bool = True,
                 template: str = "qa",
                 max_samples: Optional[int] = None):
        
        self.difficulty = difficulty
        self.data_dir = data_dir
        self.split = split
        self.include_answer = include_answer
        self.include_reasoning = include_reasoning
        self.few_shot = few_shot
        self.num_shots = num_shots
        self.seed = seed
        self.cot = cot
        self.template = template
        self.max_samples = max_samples
        
        if seed is not None:
            random.seed(seed)
        
        if self.few_shot:
            self.few_shot_prompt = self._build_few_shot_prompt()
        else:
            self.few_shot_prompt = ""
        self.dataset = self._load_dataset()
    
    def _load_dataset(self) -> Dataset:
        filename = f"{self.split}_{self.difficulty}.parquet"
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        df = pd.read_parquet(filepath)
        
        if self.max_samples and len(df) > self.max_samples:
            df = df.sample(n=self.max_samples, random_state=self.seed)
        
        processed_data = []
        for idx, row in df.iterrows():
            processed_item = self._process_example(row)
            if processed_item:
                processed_data.append(processed_item)
        
        dataset = Dataset.from_list(processed_data)
        
        print(f"Loaded {len(dataset)} examples from {self.difficulty} difficulty dataset")
        
        return dataset
    
    def _process_example(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Process a single example"""
        question = row.get('question', '')
        answer = row.get('answer', '')
        
        if not question or not answer:
            return None
        
        answer_delim = "#### "
        if answer_delim in answer:
            reasoning = answer.split(answer_delim)[0].strip()
            final_answer = answer.split(answer_delim)[-1].strip()
        else:
            match = re.search(r'([\-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$', answer)
            if match:
                final_answer = match.group(1).replace(',', '')
                reasoning = answer[:match.start()].strip()
            else:
                reasoning = answer.strip()
                final_answer = ""
        
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
            'difficulty': self.difficulty,
        }
    
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
        text = '. '.join(text.split('\n'))
        text = re.sub(r'\.+', '.', text)
        text = text.strip()
        return text
    
    def _build_few_shot_prompt(self) -> str:
        if self.difficulty == "easy":
            examples = self._get_easy_examples()
        elif self.difficulty == "medium":
            examples = self._get_medium_examples()
        else:  # hard
            examples = self._get_hard_examples()
        
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
    
    def _get_easy_examples(self) -> List[Dict[str, str]]:
        return [
            {
                'question': "John has 5 apples. He buys 3 more apples. How many apples does he have?",
                'reasoning': "John starts with 5 apples. He buys 3 more. So he has 5 + 3 = 8 apples",
                'answer': "8"
            },
        ]
    
    def _get_medium_examples(self) -> List[Dict[str, str]]:
        """Few-shot examples for medium problems"""
        return [
            {
                'question': "A store sells apples for $2 each and oranges for $3 each. If John buys 4 apples and 5 oranges, how much does he spend?",
                'reasoning': "Apples cost $2 each, so 4 apples cost 4 × 2 = $8. Oranges cost $3 each, so 5 oranges cost 5 × 3 = $15. Total cost is 8 + 15 = $23",
                'answer': "23"
            },
        ]
    
    def _get_hard_examples(self) -> List[Dict[str, str]]:
        """Few-shot examples for hard problems"""
        return [
            {
                'question': "A factory produces 240 widgets per day. Due to increased demand, they increase production by 25%. Then, due to equipment issues, they have to reduce the new production rate by 10%. How many widgets do they produce per day now?",
                'reasoning': "Initial production is 240 widgets. After 25% increase: 240 × 1.25 = 300 widgets. After 10% decrease: 300 × 0.9 = 270 widgets",
                'answer': "270"
            },
        ]
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[idx]
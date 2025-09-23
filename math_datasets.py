# math_datasets.py
import re
import random
from typing import Dict, Any, List, Optional
from datasets import load_dataset, concatenate_datasets

class MATHDataset:
    """
    MATH dataset loader - loads from EleutherAI/hendrycks_math
    The MATH dataset contains 7 subjects with problems from Level 1-5
    Total: ~7,500 training problems across all subjects
    """
    
    # Available subjects in the MATH dataset
    MATH_SUBJECTS = [
        'algebra',                    # ~1,744 problems
        'counting_and_probability',   # ~771 problems  
        'geometry',                   # ~870 problems
        'intermediate_algebra',       # ~1,295 problems
        'number_theory',              # ~869 problems
        'prealgebra',                # ~1,205 problems
        'precalculus'                # ~761 problems
    ]
    
    def __init__(self, split: str = "train", include_answer: bool = False,
                 include_reasoning: bool = True, few_shot: bool = True,
                 num_shots: int = 2, seed: Optional[int] = None,
                 cot: bool = True, template: str = "qa",
                 subjects: Optional[List[str]] = None,  # Which subjects to include
                 difficulty_level: Optional[str] = None):  # 'Level 1' to 'Level 5'
        
        self.split = split if split == "train" else "test"  # MATH only has train/test
        self.include_answer = include_answer
        self.include_reasoning = include_reasoning
        self.few_shot = few_shot
        self.num_shots = num_shots
        self.seed = seed
        self.cot = cot
        self.template = template
        self.subjects = subjects or self.MATH_SUBJECTS  # Use all subjects by default
        self.difficulty_level = difficulty_level
        
        if seed is not None:
            random.seed(seed)
        
        if self.few_shot:
            self.few_shot_prompt = self._build_few_shot_prompt()
        else:
            self.few_shot_prompt = ""
        
        self.dataset = self._load_and_process_dataset()
    
    def _extract_answer_from_solution(self, solution: str) -> str:
        """Extract final answer from MATH solution using \\boxed{...}"""
        # MATH uses \boxed{answer} format
        boxed_pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(boxed_pattern, solution)
        if matches:
            return matches[-1]  # Return last boxed answer
        
        # Fallback: look for numerical answer at the end
        num_pattern = r'([\-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$'
        match = re.search(num_pattern, solution)
        if match:
            return match.group(1).replace(',', '')
        
        return ""
    
    def _load_and_process_dataset(self):
        """Load MATH dataset from EleutherAI with proper subjects"""
        
        # Load each subject separately
        all_datasets = []
        
        for subject in self.subjects:
            try:
                print(f"Loading MATH subject: {subject}")
                subject_dataset = load_dataset(
                    "EleutherAI/hendrycks_math", 
                    subject,
                    split=self.split
                )
                
                # Add subject tag to each example
                def add_subject(example):
                    example['subject'] = subject
                    return example
                
                subject_dataset = subject_dataset.map(add_subject)
                all_datasets.append(subject_dataset)
                print(f"  Loaded {len(subject_dataset)} {subject} problems")
                
            except Exception as e:
                print(f"Warning: Could not load subject {subject}: {e}")
                continue
        
        if not all_datasets:
            raise ValueError("Could not load any MATH subjects")
        
        # Concatenate all subjects into one dataset
        dataset = concatenate_datasets(all_datasets)
        print(f"Total: Loaded {len(dataset)} MATH problems from {len(all_datasets)} subjects")
        
        # Filter by difficulty level if specified (Level 1-5)
        if self.difficulty_level:
            dataset = dataset.filter(lambda x: x.get('level') == self.difficulty_level)
            print(f"Filtered to {len(dataset)} problems at {self.difficulty_level}")
        
        def process_example(example, idx):
            question = example['problem']
            solution = example['solution']
            
            # Extract answer from solution
            final_answer = self._extract_answer_from_solution(solution)
            
            # Clean reasoning (remove boxed answer for reasoning)
            reasoning = re.sub(r'\\boxed\{[^}]*\}', '', solution).strip()
            
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
                'raw_answer': solution,
                'level': example.get('level', 'unknown'),
                'type': example.get('type', 'unknown'),
                'subject': example.get('subject', 'unknown'),
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
            example += f'    # {reasoning[:200]}\n'  # Truncate long reasoning
        
        if answer is not None:
            example += f'    return "{answer}"\n\n'
        
        return example
    
    def _clean_reasoning(self, text: str) -> str:
        # Remove LaTeX commands that might confuse the model
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\$([^\$]*)\$', r'\1', text)
        text = '. '.join(text.split('\n'))
        text = re.sub(r'\.+', '.', text)
        return text.strip()
    
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
        return [
            {
                'question': "Find the value of x if 2x + 3 = 11",
                'reasoning': "We have 2x + 3 = 11. Subtracting 3 from both sides: 2x = 8. Dividing by 2: x = 4",
                'answer': "4"
            },
            {
                'question': "What is the area of a rectangle with length 5 and width 3?",
                'reasoning': "The area of a rectangle is length times width. So area = 5 × 3 = 15",
                'answer': "15"
            },
        ]
    
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


class AQUARAT:
    """
    AQUA-RAT dataset loader - 97K algebraic word problems with rationales
    Multiple choice format with detailed explanations
    """
    
    def __init__(self, split: str = "train", include_answer: bool = False,
                 include_reasoning: bool = True, few_shot: bool = True,
                 num_shots: int = 2, seed: Optional[int] = None,
                 cot: bool = True, template: str = "qa"):
        
        self.split = split if split != "test" else "validation"  # AQUA uses validation instead of test
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
    
    def _extract_numerical_answer(self, options: List[str], correct: str) -> str:
        """Extract numerical value from the correct option"""
        if not options or not correct:
            return ""
        
        # correct is like 'A', 'B', etc.
        idx = ord(correct) - ord('A')
        if 0 <= idx < len(options):
            option_text = options[idx]
            # Extract number from option text like "A)12" or "B)$15"
            # Remove the letter and parenthesis first
            cleaned = re.sub(r'^[A-E]\)', '', option_text).strip()
            # Extract number
            match = re.search(r'([\-]?\d+(?:,\d{3})*(?:\.\d+)?)', cleaned)
            if match:
                return match.group(1).replace(',', '')
        return correct  # Return letter if no number found
    
    def _load_and_process_dataset(self):
        dataset = load_dataset("deepmind/aqua_rat", split=self.split)
        print(f"Loaded AQUA-RAT {self.split} split: {len(dataset)} problems")
        
        def process_example(example, idx):
            question = example['question']
            rationale = example['rationale']
            options = example['options']
            correct = example['correct']
            
            # Extract numerical answer
            final_answer = self._extract_numerical_answer(options, correct)
            
            # Add options to question for context
            question_with_options = question
            if options:
                question_with_options += "\nOptions: " + ", ".join(options)
            
            if self.template == 'qa':
                prompt = self._format_qa_example(question_with_options,
                                                rationale if self.include_reasoning else None,
                                                final_answer if self.include_answer else None)
            elif self.template == 'code':
                prompt = self._format_code_example(question,
                                                  rationale if self.include_reasoning else None,
                                                  final_answer if self.include_answer else None)
            else:
                prompt = f"Question: {question_with_options}\nAnswer:"
            
            if self.few_shot:
                prompt = self.few_shot_prompt + prompt
            
            return {
                'prompt': prompt,
                'final_answer': final_answer,
                'question': question,
                'raw_answer': rationale,
                'options': options,
                'correct_letter': correct,
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
            example += f'    # {reasoning[:200]}\n'
        
        if answer is not None:
            example += f'    return "{answer}"\n\n'
        
        return example
    
    def _clean_reasoning(self, text: str) -> str:
        text = '. '.join(text.split('\n'))
        text = re.sub(r'\.+', '.', text)
        return text.strip()
    
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
        return [
            {
                'question': "If a train travels 60 miles in 1.5 hours, what is its average speed?\nOptions: A)30 mph, B)40 mph, C)50 mph, D)60 mph",
                'reasoning': "Speed = Distance / Time. Speed = 60 miles / 1.5 hours = 40 mph",
                'answer': "40"
            },
            {
                'question': "A shop offers 20% discount. If an item costs $50 after discount, what was the original price?\nOptions: A)$60, B)$62.50, C)$65, D)$70",
                'reasoning': "Let original price be x. After 20% discount, price is 0.8x = 50. So x = 50/0.8 = 62.5",
                'answer': "62.5"
            },
        ]
    
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


# Aliases for backward compatibility
MATH500 = MATHDataset  # MATH500 is just another name for MATHDataset
MathDataset = MATHDataset  # Alternative capitalization
AquaRAT = AQUARAT  # Alternative naming


# Quick test if running directly
if __name__ == "__main__":
    print("Testing dataset loaders...")
    
    # Test MATH dataset
    print("\n1. Testing MATH dataset:")
    math = MATHDataset(subjects=['algebra'], split='train')
    print(f"   Algebra only: {len(math)} problems")
    
    # Test AQUA-RAT
    print("\n2. Testing AQUA-RAT dataset:")
    aqua = AQUARAT(split='train')
    print(f"   AQUA-RAT: {len(aqua)} problems")
    
    print("\nAll dataset loaders working!")
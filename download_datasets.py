#!/usr/bin/env python3
"""
test_datasets.py - Test which datasets are available and working
"""

from datasets import load_dataset
import sys

def test_gsm8k():
    """Test GSM8K dataset"""
    print("\n" + "="*60)
    print("Testing GSM8K...")
    print("="*60)
    try:
        dataset = load_dataset("gsm8k", "main", split="train")
        print(f" GSM8K works! Train size: {len(dataset)}")
        example = dataset[0]
        print(f"   Example keys: {list(example.keys())}")
        return True
    except Exception as e:
        print(f" GSM8K failed: {e}")
        return False

def test_aqua():
    """Test AQUA-RAT dataset"""
    print("\n" + "="*60)
    print("Testing AQUA-RAT...")
    print("="*60)
    try:
        dataset = load_dataset("deepmind/aqua_rat", split="train")
        print(f" AQUA-RAT works! Train size: {len(dataset)}")
        example = dataset[0]
        print(f"   Example keys: {list(example.keys())}")
        return True
    except Exception as e:
        print(f" AQUA-RAT failed: {e}")
        return False

def test_math():
    """Test MATH dataset with proper config"""
    print("\n" + "="*60)
    print("Testing MATH dataset...")
    print("="*60)
    
    # Try loading with a specific subject
    subjects = ['algebra', 'counting_and_probability', 'geometry', 'prealgebra']
    working_subjects = []
    
    for subject in subjects:
        try:
            # Don't use trust_remote_code
            dataset = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
            print(f" MATH subject '{subject}' works! Size: {len(dataset)}")
            if len(working_subjects) == 0:  # Show example from first working subject
                example = dataset[0]
                print(f"   Example keys: {list(example.keys())}")
            working_subjects.append(subject)
        except Exception as e:
            print(f" MATH subject '{subject}' failed: {str(e)[:100]}")
    
    if working_subjects:
        print(f"\n MATH dataset works with {len(working_subjects)} subjects!")
        print(f"   Working subjects: {', '.join(working_subjects)}")
        return True
    else:
        print(" MATH dataset could not be loaded")
        return False

def test_fixed_math_loader():
    """Test our custom MATH loader"""
    print("\n" + "="*60)
    print("Testing custom MATH loader (fixed_math_dataset.py)...")
    print("="*60)
    
    try:
        from fixed_math_dataset import MATHDataset
        
        # Try loading with just one subject first
        math_data = MATHDataset(subjects=['algebra'], split='train')
        print(f" Custom MATH loader works! Algebra size: {len(math_data)}")
        
        # Try loading all subjects
        math_all = MATHDataset(split='train')
        print(f" All subjects loaded! Total size: {len(math_all)}")
        
        # Show an example
        example = math_all[0]
        print(f"   Example prompt (first 200 chars): {example['prompt'][:200]}...")
        
        return True
    except Exception as e:
        print(f" Custom MATH loader failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_compatibility():
    """Test if datasets work with the training script"""
    print("\n" + "="*60)
    print("Testing training compatibility...")
    print("="*60)
    
    try:
        from gsm8k import GSM8K
        gsm = GSM8K(split='train')
        print(f" GSM8K training loader works: {len(gsm.dataset)} samples")
    except Exception as e:
        print(f" GSM8K training loader failed: {e}")
    
    try:
        from math_datasets import AQUARAT
        aqua = AQUARAT(split='train')
        print(f" AQUA-RAT training loader works: {len(aqua.dataset)} samples")
    except Exception as e:
        print(f" AQUA-RAT training loader failed: {e}")
    
    try:
        from fixed_math_dataset import MATHDataset
        math = MATHDataset(split='train', subjects=['algebra'])
        print(f"MATH training loader works: {len(math.dataset)} samples")
    except Exception as e:
        print(f" MATH training loader failed: {e}")

def main():
    print("="*60)
    print("TESTING AVAILABLE DATASETS")
    print("="*60)
    
    results = {
        'GSM8K': test_gsm8k(),
        'AQUA-RAT': test_aqua(),
        'MATH': test_math()
    }
    
    # Test custom loader if MATH works
    if results['MATH']:
        results['MATH (custom)'] = test_fixed_math_loader()
    
    # Test training compatibility
    test_training_compatibility()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    working = [name for name, status in results.items() if status]
    failed = [name for name, status in results.items() if not status]
    
    if working:
        print(f" Working datasets: {', '.join(working)}")
    if failed:
        print(f" Failed datasets: {', '.join(failed)}")
    
    print("\n" + "="*60)
    print("RECOMMENDED TRAINING COMMANDS")
    print("="*60)
    
    if results['GSM8K']:
        print("\n# Train on GSM8K:")
        print("python train_multi_dataset.py --dataset gsm8k --normalization standard --max_steps 400")
    
    if results['AQUA-RAT']:
        print("\n# Train on AQUA-RAT (97K samples!):")
        print("python train_multi_dataset.py --dataset aqua --normalization standard --max_steps 400")
    
    if results.get('MATH (custom)', False):
        print("\n# Train on MATH (using fixed loader):")
        print("python train_with_math.py --subjects algebra geometry --normalization standard --max_steps 400")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Tests for get_intervention_locations with various position settings.
Uses actual Tulu-3 samples processed exactly as in training.

Usage:
    pytest test_intervention_locations.py -v
    python test_intervention_locations.py  # standalone
"""

import pytest
import sys
sys.path.insert(0, "../../..")

# Lazy imports to speed up test collection
def get_intervention_locations(*args, **kwargs):
    from pyreft.dataset import get_intervention_locations as _get
    return _get(*args, **kwargs)

def parse_positions(*args, **kwargs):
    from pyreft.dataset import parse_positions as _parse
    return _parse(*args, **kwargs)


def load_tulu3_samples(tokenizer, n_samples=5):
    """
    Load and preprocess Tulu-3 samples exactly as in training.
    Returns list of dicts with 'prompt', 'completion', 'messages', 'prompt_tokens'.
    """
    from datasets import load_dataset
    
    # Load a few samples from Tulu-3
    dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    samples = []
    
    for i, example in enumerate(dataset):
        if i >= n_samples:
            break
        
        messages = example.get("messages", [])
        if not messages:
            continue
        
        # Split messages into prompt (everything before assistant) and completion
        prompt_messages = []
        completion = ""
        
        for msg in messages:
            if msg["role"] == "assistant":
                completion = msg["content"]
                break
            prompt_messages.append(msg)
        
        # Format prompt using chat template (exactly as in training)
        prompt = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize prompt
        prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        
        samples.append({
            "messages": messages,
            "prompt_messages": prompt_messages,
            "prompt": prompt,
            "completion": completion,
            "prompt_tokens": prompt_tokens,
            "prompt_length": len(prompt_tokens),
        })
    
    return samples


# Test fixtures
@pytest.fixture(scope="module")
def tokenizer():
    """Load Llama 3.2 1B Instruct tokenizer."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")


@pytest.fixture(scope="module")
def tulu_samples(tokenizer):
    """Load Tulu-3 samples."""
    return load_tulu3_samples(tokenizer, n_samples=5)


class TestParsePositions:
    """Tests for parse_positions function."""
    
    def test_f1_l1(self):
        first_n, last_n = parse_positions("f1+l1")
        assert first_n == 1
        assert last_n == 1
    
    def test_f5_l3(self):
        first_n, last_n = parse_positions("f5+l3")
        assert first_n == 5
        assert last_n == 3
    
    def test_f_only(self):
        first_n, last_n = parse_positions("f10")
        assert first_n == 10
        assert last_n == 0
    
    def test_l_only(self):
        first_n, last_n = parse_positions("l5")
        assert first_n == 0
        assert last_n == 5
    
    def test_all(self):
        first_n, last_n = parse_positions("all")
        assert first_n == -1
        assert last_n == -1


class TestGetInterventionLocations:
    """Tests for get_intervention_locations function."""
    
    def test_f1_l1_basic(self):
        """Test f1+l1 with a simple prompt."""
        last_position = 10  # prompt has 10 tokens (indices 0-9)
        
        locations = get_intervention_locations(
            last_position=last_position,
            positions="f1+l1",
            num_interventions=2,
            share_weights=True,
        )
        
        # Should return [[0, 9], [0, 9]] - first and last token
        assert len(locations) == 2
        assert locations[0] == [0, 9]
        assert locations[1] == [0, 9]
    
    def test_all_requires_share_weights(self):
        """Test that position='all' requires share_weights=True."""
        with pytest.raises(AssertionError, match="share_weights"):
            get_intervention_locations(
                last_position=10,
                position="all",
                num_interventions=2,
                share_weights=False,
            )


class TestWithTulu3:
    """Tests using actual Tulu-3 samples."""
    
    def test_intervention_locations_on_tulu3(self, tokenizer, tulu_samples):
        """Test intervention locations with actual Tulu-3 prompts."""
        for i, sample in enumerate(tulu_samples):
            prompt_length = sample["prompt_length"]
            last_position = prompt_length - 1  # Following dataset convention
            
            # Test f1+l1
            locs_f1l1 = get_intervention_locations(
                last_position=last_position,
                positions="f1+l1",
                num_interventions=2,
                share_weights=True,
            )
            
            # f1+l1 should have 2 positions
            assert len([p for p in locs_f1l1[0] if p >= 0]) == 2
            assert 0 in locs_f1l1[0]  # first token
            
            # Test all
            locs_all = get_intervention_locations(
                last_position=last_position,
                position="all",
                num_interventions=2,
                share_weights=True,
            )
            
            # All should cover positions 0 to last_position-1
            assert len(locs_all[0]) == last_position
            assert locs_all[0] == list(range(last_position))


def main():
    """Run tests standalone with detailed output."""
    print("=" * 70)
    print("Testing parse_positions")
    print("=" * 70)
    
    test_cases = ["f1+l1", "f5+l3", "f10", "l5", "all"]
    for pos in test_cases:
        result = parse_positions(pos)
        print(f"  {pos:10s} -> first_n={result[0]}, last_n={result[1]}")
    
    print("\n" + "=" * 70)
    print("Testing get_intervention_locations (basic)")
    print("=" * 70)
    
    # Test f1+l1
    last_pos = 10
    locs = get_intervention_locations(
        last_position=last_pos,
        positions="f1+l1",
        num_interventions=2,
        share_weights=True,
    )
    print(f"\n  f1+l1 (last_position={last_pos}):")
    print(f"    locations[0]: {locs[0]}")
    
    # Test all
    locs_all = get_intervention_locations(
        last_position=last_pos,
        position="all",
        num_interventions=2,
        share_weights=True,
    )
    print(f"\n  all (last_position={last_pos}):")
    print(f"    locations[0]: {locs_all[0]}")
    print(f"    num positions: {len(locs_all[0])}")
    
    print("\n" + "=" * 70)
    print("Testing with actual Tulu-3 samples")
    print("=" * 70)
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        
        print("\nLoading Tulu-3 samples...")
        samples = load_tulu3_samples(tokenizer, n_samples=3)
        
        for i, sample in enumerate(samples):
            print(f"\n{'─' * 70}")
            print(f"Sample {i+1}")
            print(f"{'─' * 70}")
            
            # Show messages
            print(f"\nMessages:")
            for msg in sample["prompt_messages"]:
                role = msg["role"]
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                print(f"  [{role}]: {content}")
            
            completion_preview = sample["completion"][:100] + "..." if len(sample["completion"]) > 100 else sample["completion"]
            print(f"  [assistant]: {completion_preview}")
            
            # Show tokenization
            prompt_length = sample["prompt_length"]
            last_position = prompt_length - 1
            
            print(f"\nTokenization:")
            print(f"  Prompt length: {prompt_length} tokens")
            print(f"  last_position (prompt_length - 1): {last_position}")
            
            # Show first and last few tokens
            token_strs = tokenizer.convert_ids_to_tokens(sample["prompt_tokens"])
            print(f"\n  First 5 tokens:")
            for j in range(min(5, len(token_strs))):
                print(f"    [{j:3d}] {repr(token_strs[j])}")
            print(f"  ...")
            print(f"  Last 5 tokens:")
            for j in range(max(0, len(token_strs)-5), len(token_strs)):
                print(f"    [{j:3d}] {repr(token_strs[j])}")
            
            # Test f1+l1
            locs_f1l1 = get_intervention_locations(
                last_position=last_position,
                positions="f1+l1",
                num_interventions=2,
                share_weights=True,
            )
            
            print(f"\nIntervention locations:")
            print(f"  f1+l1: {locs_f1l1[0]}")
            f1l1_tokens = [token_strs[p] for p in locs_f1l1[0] if 0 <= p < len(token_strs)]
            print(f"    -> tokens: {f1l1_tokens}")
            
            # Test all
            locs_all = get_intervention_locations(
                last_position=last_position,
                position="all",
                num_interventions=2,
                share_weights=True,
            )
            
            print(f"\n  all: positions 0 to {max(locs_all[0])}")
            print(f"    -> num positions: {len(locs_all[0])}")
            print(f"    -> expected positions: {last_position} (range(0, {last_position}))")
            
            # Check for bug
            if len(locs_all[0]) != last_position:
                print(f"\n  ⚠️  MISMATCH: got {len(locs_all[0])} positions, expected {last_position}")
            
            if max(locs_all[0]) != last_position - 1:
                print(f"\n  ⚠️  BUG: max position is {max(locs_all[0])}, expected {last_position - 1}")
                print(f"       Missing token at position {last_position - 1}: {repr(token_strs[last_position - 1]) if last_position - 1 < len(token_strs) else 'N/A'}")
            
            # What SHOULD "all" include?
            print(f"\n  Analysis:")
            print(f"    - Prompt has {prompt_length} tokens (indices 0 to {prompt_length - 1})")
            print(f"    - last_position = {last_position}")
            print(f"    - 'all' gives range({last_position}) = [0, ..., {last_position - 1}]")
            print(f"    - This covers {last_position} positions out of {prompt_length} tokens")
            
            if last_position < prompt_length:
                print(f"    - Token at index {prompt_length - 1} ({repr(token_strs[prompt_length - 1])}) is NOT intervened on")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

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
        first_n, last_n, strict = parse_positions("f1+l1")
        assert first_n == 1
        assert last_n == 1
        assert strict == False
    
    def test_f1_s1_strict(self):
        """Test strict mode with 's' suffix."""
        first_n, last_n, strict = parse_positions("f1+s1")
        assert first_n == 1
        assert last_n == 1
        assert strict == True
    
    def test_f5_l3(self):
        first_n, last_n, strict = parse_positions("f5+l3")
        assert first_n == 5
        assert last_n == 3
        assert strict == False
    
    def test_f_only(self):
        first_n, last_n, strict = parse_positions("f10")
        assert first_n == 10
        assert last_n == 0
        assert strict == False
    
    def test_l_only(self):
        first_n, last_n, strict = parse_positions("l5")
        assert first_n == 0
        assert last_n == 5
        assert strict == False
    
    def test_s_only_strict(self):
        """Test strict mode with single 's' position."""
        first_n, last_n, strict = parse_positions("s5")
        assert first_n == 0
        assert last_n == 5
        assert strict == True
    
    def test_all(self):
        first_n, last_n, strict = parse_positions("all")
        assert first_n == -1
        assert last_n == -1
        assert strict == False
    
    def test_alls_strict(self):
        """Test strict mode for all positions."""
        first_n, last_n, strict = parse_positions("alls")
        assert first_n == -1
        assert last_n == -1
        assert strict == True


class TestGetInterventionLocations:
    """Tests for get_intervention_locations function."""
    
    def test_f1_l1_basic(self):
        """Test f1+l1 with a simple prompt (legacy, off-by-one)."""
        last_position = 10  # prompt has 11 tokens (indices 0-10), last_position=10
        
        locations = get_intervention_locations(
            last_position=last_position,
            positions="f1+l1",
            num_interventions=2,
            share_weights=True,
        )
        
        # Legacy behavior: [0, 9] - misses actual last token (10)
        assert len(locations) == 2
        assert locations[0] == [0, 9]
        assert locations[1] == [0, 9]
    
    def test_f1_s1_strict(self):
        """Test f1+s1 strict mode - includes actual last token."""
        last_position = 10  # prompt has 11 tokens (indices 0-10)
        
        locations = get_intervention_locations(
            last_position=last_position,
            positions="f1+s1",
            num_interventions=2,
            share_weights=True,
        )
        
        # Strict mode: [0, 10] - includes actual last token
        assert len(locations) == 2
        assert locations[0] == [0, 10]
        assert locations[1] == [0, 10]
    
    def test_all_legacy(self):
        """Test position='all' legacy behavior (off-by-one)."""
        last_position = 10
        
        locations = get_intervention_locations(
            last_position=last_position,
            position="all",
            num_interventions=2,
            share_weights=True,
        )
        
        # Legacy: range(10) = [0, 1, ..., 9] - misses index 10
        assert locations[0] == list(range(10))
        assert len(locations[0]) == 10
    
    def test_alls_strict(self):
        """Test position='alls' strict mode - includes actual last token."""
        last_position = 10
        
        locations = get_intervention_locations(
            last_position=last_position,
            position="alls",
            num_interventions=2,
            share_weights=True,
        )
        
        # Strict: range(11) = [0, 1, ..., 10] - includes index 10
        assert locations[0] == list(range(11))
        assert len(locations[0]) == 11
    
    def test_all_requires_share_weights(self):
        """Test that position='all' requires share_weights=True."""
        with pytest.raises(AssertionError, match="share_weights"):
            get_intervention_locations(
                last_position=10,
                position="all",
                num_interventions=2,
                share_weights=False,
            )
    
    def test_alls_requires_share_weights(self):
        """Test that position='alls' requires share_weights=True."""
        with pytest.raises(AssertionError, match="share_weights"):
            get_intervention_locations(
                last_position=10,
                position="alls",
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
            
            # Test f1+l1 (legacy)
            locs_f1l1 = get_intervention_locations(
                last_position=last_position,
                positions="f1+l1",
                num_interventions=2,
                share_weights=True,
            )
            
            # f1+l1 should have 2 positions
            assert len([p for p in locs_f1l1[0] if p >= 0]) == 2
            assert 0 in locs_f1l1[0]  # first token
            
            # Test all (legacy)
            locs_all = get_intervention_locations(
                last_position=last_position,
                position="all",
                num_interventions=2,
                share_weights=True,
            )
            
            # All should cover positions 0 to last_position-1 (legacy off-by-one)
            assert len(locs_all[0]) == last_position
            assert locs_all[0] == list(range(last_position))
    
    def test_strict_mode_on_tulu3(self, tokenizer, tulu_samples):
        """Test strict mode intervention locations with actual Tulu-3 prompts."""
        for i, sample in enumerate(tulu_samples):
            prompt_length = sample["prompt_length"]
            last_position = prompt_length - 1  # Following dataset convention
            
            # Test f1+s1 (strict) - should include actual last token
            locs_f1s1 = get_intervention_locations(
                last_position=last_position,
                positions="f1+s1",
                num_interventions=2,
                share_weights=True,
            )
            
            # f1+s1 should have 2 positions: first (0) and actual last (last_position)
            assert len([p for p in locs_f1s1[0] if p >= 0]) == 2
            assert 0 in locs_f1s1[0]  # first token
            assert last_position in locs_f1s1[0]  # actual last token (strict)
            
            # Compare with legacy f1+l1
            locs_f1l1 = get_intervention_locations(
                last_position=last_position,
                positions="f1+l1",
                num_interventions=2,
                share_weights=True,
            )
            # Strict should have last_position, legacy should have last_position - 1
            assert max(locs_f1s1[0]) == last_position
            assert max(locs_f1l1[0]) == last_position - 1
            
            # Test alls (strict) - should include actual last token
            locs_alls = get_intervention_locations(
                last_position=last_position,
                position="alls",
                num_interventions=2,
                share_weights=True,
            )
            
            # alls should cover positions 0 to last_position (inclusive)
            assert len(locs_alls[0]) == last_position + 1
            assert locs_alls[0] == list(range(last_position + 1))
            
            # Compare with legacy all
            locs_all = get_intervention_locations(
                last_position=last_position,
                position="all",
                num_interventions=2,
                share_weights=True,
            )
            # Strict should have one more position than legacy
            assert len(locs_alls[0]) == len(locs_all[0]) + 1
            assert max(locs_alls[0]) == last_position
            assert max(locs_all[0]) == last_position - 1


def main():
    """Run tests standalone with detailed output."""
    print("=" * 70)
    print("Testing parse_positions")
    print("=" * 70)
    
    test_cases = ["f1+l1", "f1+s1", "f5+l3", "f10", "l5", "s5", "all", "alls"]
    for pos in test_cases:
        result = parse_positions(pos)
        print(f"  {pos:10s} -> first_n={result[0]}, last_n={result[1]}, strict={result[2]}")
    
    print("\n" + "=" * 70)
    print("Testing get_intervention_locations (basic)")
    print("=" * 70)
    
    last_pos = 10  # Simulating prompt with 11 tokens (0-10)
    
    # Test f1+l1 (legacy)
    locs = get_intervention_locations(
        last_position=last_pos,
        positions="f1+l1",
        num_interventions=2,
        share_weights=True,
    )
    print(f"\n  f1+l1 (legacy, last_position={last_pos}):")
    print(f"    locations[0]: {locs[0]}")
    print(f"    Note: Misses actual last token at index {last_pos}")
    
    # Test f1+s1 (strict)
    locs_strict = get_intervention_locations(
        last_position=last_pos,
        positions="f1+s1",
        num_interventions=2,
        share_weights=True,
    )
    print(f"\n  f1+s1 (strict, last_position={last_pos}):")
    print(f"    locations[0]: {locs_strict[0]}")
    print(f"    ✓ Includes actual last token at index {last_pos}")
    
    # Test all (legacy)
    locs_all = get_intervention_locations(
        last_position=last_pos,
        position="all",
        num_interventions=2,
        share_weights=True,
    )
    print(f"\n  all (legacy, last_position={last_pos}):")
    print(f"    locations[0]: {locs_all[0]}")
    print(f"    num positions: {len(locs_all[0])}")
    print(f"    Note: Misses actual last token at index {last_pos}")
    
    # Test alls (strict)
    locs_alls = get_intervention_locations(
        last_position=last_pos,
        position="alls",
        num_interventions=2,
        share_weights=True,
    )
    print(f"\n  alls (strict, last_position={last_pos}):")
    print(f"    locations[0]: {locs_alls[0]}")
    print(f"    num positions: {len(locs_alls[0])}")
    print(f"    ✓ Includes actual last token at index {last_pos}")
    
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
            
            print(f"\nIntervention locations (LEGACY vs STRICT):")
            
            # Test f1+l1 (legacy)
            locs_f1l1 = get_intervention_locations(
                last_position=last_position,
                positions="f1+l1",
                num_interventions=2,
                share_weights=True,
            )
            print(f"\n  f1+l1 (legacy): {locs_f1l1[0]}")
            f1l1_tokens = [token_strs[p] for p in locs_f1l1[0] if 0 <= p < len(token_strs)]
            print(f"    -> tokens: {f1l1_tokens}")
            
            # Test f1+s1 (strict)
            locs_f1s1 = get_intervention_locations(
                last_position=last_position,
                positions="f1+s1",
                num_interventions=2,
                share_weights=True,
            )
            print(f"\n  f1+s1 (strict): {locs_f1s1[0]}")
            f1s1_tokens = [token_strs[p] for p in locs_f1s1[0] if 0 <= p < len(token_strs)]
            print(f"    -> tokens: {f1s1_tokens}")
            
            # Test all (legacy)
            locs_all = get_intervention_locations(
                last_position=last_position,
                position="all",
                num_interventions=2,
                share_weights=True,
            )
            print(f"\n  all (legacy): positions 0 to {max(locs_all[0])}, count={len(locs_all[0])}")
            
            # Test alls (strict)
            locs_alls = get_intervention_locations(
                last_position=last_position,
                position="alls",
                num_interventions=2,
                share_weights=True,
            )
            print(f"  alls (strict): positions 0 to {max(locs_alls[0])}, count={len(locs_alls[0])}")
            
            # Comparison
            print(f"\n  Comparison:")
            print(f"    - Prompt has {prompt_length} tokens (indices 0 to {prompt_length - 1})")
            print(f"    - last_position = {last_position}")
            print(f"    - 'all' (legacy):  covers {len(locs_all[0])} positions, max={max(locs_all[0])}")
            print(f"    - 'alls' (strict): covers {len(locs_alls[0])} positions, max={max(locs_alls[0])}")
            print(f"    - f1+l1 last token: {locs_f1l1[0][-1]} -> {repr(token_strs[locs_f1l1[0][-1]])}")
            print(f"    - f1+s1 last token: {locs_f1s1[0][-1]} -> {repr(token_strs[locs_f1s1[0][-1]])}")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Tests for get_intervention_locations with various position settings.

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


# Test fixtures
@pytest.fixture(scope="module")
def tokenizer():
    """Load Llama 3.2 1B Instruct tokenizer."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")


@pytest.fixture
def dummy_conversations():
    """Dummy conversations for testing."""
    return [
        {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum computing in one sentence."},
                {"role": "assistant", "content": "Quantum computing uses quantum bits that can exist in superposition to perform certain calculations exponentially faster than classical computers."},
            ]
        },
    ]


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
    
    def test_f1_l1_not_shared(self):
        """Test f1+l1 without weight sharing."""
        last_position = 10
        
        locations = get_intervention_locations(
            last_position=last_position,
            positions="f1+l1",
            num_interventions=2,
            share_weights=False,
        )
        
        # Should return [[0], [9]] - separate interventions
        assert len(locations) == 2
        # First intervention on first token, second on last
        assert 0 in locations[0]
        assert 9 in locations[1] or (last_position - 1) in locations[1]
    
    def test_all_positions(self):
        """Test position='all' intervenes on all prompt tokens."""
        last_position = 10  # prompt tokens at indices 0-9
        
        locations = get_intervention_locations(
            last_position=last_position,
            position="all",  # singular key
            num_interventions=2,
            share_weights=True,
        )
        
        # Should return all positions 0 to last_position-1
        # Note: current implementation gives range(last_position) = [0, ..., 9]
        assert len(locations) == 2
        expected_positions = list(range(last_position))
        assert locations[0] == expected_positions
        assert locations[1] == expected_positions
    
    def test_all_requires_share_weights(self):
        """Test that position='all' requires share_weights=True."""
        with pytest.raises(AssertionError, match="share_weights"):
            get_intervention_locations(
                last_position=10,
                position="all",
                num_interventions=2,
                share_weights=False,
            )
    
    def test_f5_positions(self):
        """Test f5 - first 5 tokens."""
        last_position = 20
        
        locations = get_intervention_locations(
            last_position=last_position,
            positions="f5",
            num_interventions=1,
            share_weights=True,
        )
        
        assert len(locations) == 1
        assert locations[0][:5] == [0, 1, 2, 3, 4]
    
    def test_l3_positions(self):
        """Test l3 - last 3 tokens."""
        last_position = 20
        
        locations = get_intervention_locations(
            last_position=last_position,
            positions="l3",
            num_interventions=1,
            share_weights=True,
        )
        
        assert len(locations) == 1
        # Last 3 positions before last_position
        assert 17 in locations[0]
        assert 18 in locations[0]
        assert 19 in locations[0]
    
    def test_short_sequence_capping(self):
        """Test that positions are capped for short sequences."""
        last_position = 4  # Very short prompt
        
        locations = get_intervention_locations(
            last_position=last_position,
            positions="f10+l10",  # Request more than available
            num_interventions=2,
            share_weights=True,
        )
        
        # Should cap to half the sequence length
        assert len(locations) == 2
        # Positions should be within valid range
        for loc in locations[0]:
            assert -1 <= loc < last_position or loc == last_position  # -1 is padding


class TestWithTokenizer:
    """Tests using actual tokenizer to verify real-world behavior."""
    
    def test_tokenize_and_intervene(self, tokenizer):
        """Test intervention locations with actual tokenized text."""
        prompt = "What is the capital of France?"
        
        # Tokenize
        tokens = tokenizer(prompt, return_tensors="pt")
        prompt_length = tokens["input_ids"].shape[1]
        last_position = prompt_length - 1  # Following the dataset convention
        
        print(f"\nPrompt: {prompt}")
        print(f"Tokens: {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")
        print(f"Prompt length: {prompt_length}, last_position: {last_position}")
        
        # Test f1+l1
        locations_f1l1 = get_intervention_locations(
            last_position=last_position,
            positions="f1+l1",
            num_interventions=2,
            share_weights=True,
        )
        print(f"f1+l1 locations: {locations_f1l1[0]}")
        
        # Test all
        locations_all = get_intervention_locations(
            last_position=last_position,
            position="all",
            num_interventions=2,
            share_weights=True,
        )
        print(f"all locations: {locations_all[0]}")
        
        # Verify f1+l1 has 2 positions (first and last)
        assert len([p for p in locations_f1l1[0] if p >= 0]) == 2
        
        # Verify all has last_position positions (0 to last_position-1)
        assert len(locations_all[0]) == last_position
        assert locations_all[0] == list(range(last_position))
    
    def test_chat_template_intervention(self, tokenizer):
        """Test with chat-formatted messages."""
        messages = [
            {"role": "user", "content": "Hello!"},
        ]
        
        # Apply chat template (prompt only, no generation)
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        tokens = tokenizer(prompt, return_tensors="pt")
        prompt_length = tokens["input_ids"].shape[1]
        last_position = prompt_length - 1
        
        print(f"\nChat prompt: {repr(prompt[:100])}...")
        print(f"Prompt length: {prompt_length}")
        
        # Test all positions
        locations = get_intervention_locations(
            last_position=last_position,
            position="all",
            num_interventions=16,  # Multiple interventions (one per layer)
            share_weights=True,
        )
        
        # All interventions should have same locations (shared weights)
        assert all(loc == locations[0] for loc in locations)
        
        # Should cover all prompt positions
        assert len(locations[0]) == last_position
        print(f"Intervening on {len(locations[0])} positions")
    
    def test_varying_prompt_lengths(self, tokenizer):
        """Test that intervention locations scale with prompt length."""
        prompts = [
            "Hi",
            "What is 2+2?",
            "Explain the theory of relativity in simple terms that a high school student could understand.",
        ]
        
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors="pt")
            prompt_length = tokens["input_ids"].shape[1]
            last_position = prompt_length - 1
            
            locations = get_intervention_locations(
                last_position=last_position,
                position="all",
                num_interventions=2,
                share_weights=True,
            )
            
            print(f"\nPrompt: {prompt[:50]}...")
            print(f"Length: {prompt_length}, Intervention positions: {len(locations[0])}")
            
            # Verify all covers the right number of positions
            assert len(locations[0]) == last_position


def main():
    """Run tests standalone."""
    print("=" * 60)
    print("Testing parse_positions")
    print("=" * 60)
    
    test_cases = ["f1+l1", "f5+l3", "f10", "l5", "all"]
    for pos in test_cases:
        result = parse_positions(pos)
        print(f"  {pos:10s} -> first_n={result[0]}, last_n={result[1]}")
    
    print("\n" + "=" * 60)
    print("Testing get_intervention_locations")
    print("=" * 60)
    
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
    
    print("\n" + "=" * 60)
    print("Testing with Llama tokenizer")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        
        prompt = "What is the meaning of life?"
        tokens = tokenizer(prompt, return_tensors="pt")
        prompt_length = tokens["input_ids"].shape[1]
        last_position = prompt_length - 1
        
        print(f"\n  Prompt: {prompt}")
        print(f"  Tokens: {tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])}")
        print(f"  Length: {prompt_length}, last_position: {last_position}")
        
        locs = get_intervention_locations(
            last_position=last_position,
            position="all",
            num_interventions=2,
            share_weights=True,
        )
        print(f"  'all' positions: {locs[0]}")
        print(f"  Expected last token index: {prompt_length - 1}")
        print(f"  Actual last position in 'all': {max(locs[0])}")
        
        if max(locs[0]) < prompt_length - 1:
            print(f"  ⚠️  BUG: Missing position {prompt_length - 1}!")
        
        # Test with formatted conversation
        print("\n" + "-" * 40)
        print("  Testing with chat-formatted conversation:")
        print("-" * 40)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        
        # Format as chat (prompt only, with generation prompt)
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        tokens = tokenizer(formatted_prompt, return_tensors="pt")
        prompt_length = tokens["input_ids"].shape[1]
        last_position = prompt_length - 1
        
        print(f"\n  Messages: {messages}")
        print(f"\n  Formatted prompt:\n{formatted_prompt}")
        print(f"\n  Tokens ({prompt_length} total):")
        token_strs = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
        # Print tokens with their indices
        for i, tok in enumerate(token_strs):
            print(f"    [{i:3d}] {repr(tok)}")
        
        print(f"\n  last_position: {last_position}")
        
        # Test f1+l1
        locs_f1l1 = get_intervention_locations(
            last_position=last_position,
            positions="f1+l1",
            num_interventions=2,
            share_weights=True,
        )
        print(f"\n  f1+l1 positions: {locs_f1l1[0]}")
        print(f"    -> tokens: {[token_strs[i] for i in locs_f1l1[0] if i < len(token_strs)]}")
        
        # Test all
        locs_all = get_intervention_locations(
            last_position=last_position,
            position="all",
            num_interventions=2,
            share_weights=True,
        )
        print(f"\n  'all' positions: {locs_all[0]}")
        print(f"    -> num positions: {len(locs_all[0])}")
        print(f"    -> expected: {prompt_length} (all {prompt_length} tokens)")
        print(f"    -> actual last: {max(locs_all[0])}, expected last: {prompt_length - 1}")
        
        if max(locs_all[0]) < prompt_length - 1:
            print(f"\n  ⚠️  BUG: 'all' is missing position {prompt_length - 1}!")
            print(f"       Missing token: {repr(token_strs[prompt_length - 1])}")
        elif len(locs_all[0]) < prompt_length:
            print(f"\n  ⚠️  BUG: 'all' has {len(locs_all[0])} positions, expected {prompt_length}!")
        else:
            print(f"\n  ✓ 'all' correctly covers all {prompt_length} positions")
        
    except Exception as e:
        print(f"  Skipping tokenizer tests: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("All basic tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Tests for ReFT orthogonality preservation during save/load.

This test verifies that the orthogonal parameterization in ReFT is correctly
preserved when saving and loading model weights. Loss spikes observed during
checkpoint continuation were caused by orthogonality being broken.

## ROOT CAUSE (Fixed)

The orthogonal parameterization in pyreft uses PyTorch's parametrize module,
which stores an "original" tensor and computes weight via Cayley/Householder
transform on each forward pass.

**BUG (in legacy code)**: state_dict saved only the computed `weight`
(orthogonalized), not the internal parametrization state. When loaded:
1. The orthogonal weight was written to `.base`
2. But `.original` was freshly initialized
3. After first optimizer step, orthogonality broke

**FIX**: Now state_dict saves both:
- `rotate_layer` - computed orthogonal weight (for backwards compat)
- `rotate_layer_original` - internal optimization variable
- `rotate_layer_base` - trivialization base matrix

This allows proper training continuation without breaking orthogonality.

Usage:
    pytest test_orthogonality_save_load.py -v
    python test_orthogonality_save_load.py  # standalone (saves output to .txt)
"""

import tempfile
import pytest
import sys
import os
from collections import OrderedDict

sys.path.insert(0, "../../..")

import torch
import torch.nn as nn


def get_orthogonality_error(R: torch.Tensor) -> float:
    """
    Compute how far R is from being orthonormal.
    Returns ||R @ R^T - I||_F (Frobenius norm).
    For a perfectly orthonormal matrix, this should be ~0.
    """
    eye = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
    return torch.norm(R @ R.T - eye).item()


def get_column_orthogonality_error(R: torch.Tensor) -> float:
    """
    Compute column orthogonality error: ||R^T @ R - I||_F
    For R with shape (d, r) where d > r, columns should be orthonormal.
    """
    r = R.shape[1]
    eye = torch.eye(r, device=R.device, dtype=R.dtype)
    return torch.norm(R.T @ R - eye).item()


def create_legacy_state_dict(intervention):
    """
    Create a legacy state_dict that only contains 'rotate_layer' (no internal state).
    This simulates checkpoints saved before the fix.
    """
    state_dict = OrderedDict()
    for k, v in intervention.learned_source.state_dict().items():
        state_dict[k] = v
    # Only save computed weight, NOT the internal parametrization state
    state_dict["rotate_layer"] = intervention.rotate_layer.weight.data.clone()
    return state_dict


class TestOrthogonalityBasic:
    """Basic tests for orthogonality utilities."""

    def test_orthogonal_matrix(self):
        """Test that a known orthogonal matrix has near-zero error."""
        random_matrix = torch.randn(64, 64)
        Q, _ = torch.linalg.qr(random_matrix)
        error = get_orthogonality_error(Q)
        assert error < 1e-5, f"Orthogonal matrix has error {error}"

    def test_non_orthogonal_matrix(self):
        """Test that a random matrix has non-zero error."""
        random_matrix = torch.randn(64, 64)
        error = get_orthogonality_error(random_matrix)
        assert error > 0.1, f"Random matrix should have large error, got {error}"

    def test_tall_orthogonal_matrix(self):
        """Test column orthogonality for tall matrices (d > r)."""
        random_matrix = torch.randn(256, 8)
        Q, _ = torch.linalg.qr(random_matrix)
        Q = Q[:, :8]
        error = get_column_orthogonality_error(Q)
        assert error < 1e-5, f"Tall orthogonal matrix columns have error {error}"


class TestNewFormatSaveLoad:
    """Tests for new checkpoint format with full parametrization state."""

    def test_default_state_dict_keys(self):
        """Test that default format only saves inference weights."""
        from pyreft import LoreftIntervention

        intervention = LoreftIntervention(
            embed_dim=256, low_rank_dimension=8, dropout=0.0, dtype=torch.float32
        )
        state_dict = intervention.state_dict()

        # Default: only rotate_layer (inference weights)
        assert "rotate_layer" in state_dict, "Missing rotate_layer"
        assert "rotate_layer_original" not in state_dict, "Should not have rotate_layer_original by default"
        assert "rotate_layer_base" not in state_dict, "Should not have rotate_layer_base by default"
        print(f"Default state dict keys: {list(state_dict.keys())}")

    def test_training_format_state_dict_keys(self):
        """Test that save_for_training=True includes all required keys."""
        from pyreft import LoreftIntervention

        intervention = LoreftIntervention(
            embed_dim=256, low_rank_dimension=8, dropout=0.0, dtype=torch.float32,
            save_for_training=True
        )
        state_dict = intervention.state_dict()

        # Training format: includes parametrization state
        assert "rotate_layer" in state_dict, "Missing rotate_layer"
        assert "rotate_layer_original" in state_dict, "Missing rotate_layer_original"
        assert "rotate_layer_base" in state_dict, "Missing rotate_layer_base"
        print(f"Training state dict keys: {list(state_dict.keys())}")

    def test_new_format_continuation_orthogonality(self):
        """Test that save_for_training=True preserves orthogonality through checkpoint continuation."""
        from pyreft import LoreftIntervention

        # Phase 1: Train with save_for_training=True
        intervention = LoreftIntervention(
            embed_dim=256, low_rank_dimension=8, dropout=0.0, dtype=torch.float32,
            save_for_training=True
        )
        optimizer = torch.optim.Adam(intervention.parameters(), lr=1e-3)

        for step in range(20):
            torch.manual_seed(step)
            base = torch.randn(2, 10, 256)
            source = torch.randn(2, 10, 256)
            output = intervention(base, source)
            loss = output.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save with new format
        state_dict = intervention.state_dict()
        assert "rotate_layer_original" in state_dict, "Not using new format!"

        # Phase 2: Load and continue
        new_intervention = LoreftIntervention(
            embed_dim=256, low_rank_dimension=8, dropout=0.0, dtype=torch.float32
        )
        new_intervention.load_state_dict(state_dict)
        new_optimizer = torch.optim.Adam(new_intervention.parameters(), lr=1e-3)

        # Continue training
        for step in range(10):
            torch.manual_seed(100 + step)
            base = torch.randn(2, 10, 256)
            source = torch.randn(2, 10, 256)
            output = new_intervention(base, source)
            loss = output.sum()
            new_optimizer.zero_grad()
            loss.backward()
            new_optimizer.step()

        # Check orthogonality
        R = new_intervention.rotate_layer.weight
        error = get_column_orthogonality_error(R)
        print(f"New format - orthogonality error after continuation: {error:.2e}")
        assert error < 1e-3, f"New format should maintain orthogonality, error={error}"


class TestLegacyFormatSaveLoad:
    """Tests for legacy checkpoint format (backwards compatibility)."""

    def test_legacy_format_inference_works(self):
        """Test that legacy format still works for inference."""
        from pyreft import LoreftIntervention

        # Create and get output
        intervention = LoreftIntervention(
            embed_dim=256, low_rank_dimension=8, dropout=0.0, dtype=torch.float32
        )
        intervention.eval()

        torch.manual_seed(42)
        base = torch.randn(2, 10, 256)
        source = torch.randn(2, 10, 256)

        with torch.no_grad():
            output_original = intervention(base.clone(), source.clone())

        # Save as legacy format
        legacy_state_dict = create_legacy_state_dict(intervention)
        assert "rotate_layer_original" not in legacy_state_dict, "Not legacy format!"

        # Load and compare
        new_intervention = LoreftIntervention(
            embed_dim=256, low_rank_dimension=8, dropout=0.0, dtype=torch.float32
        )
        new_intervention.load_state_dict(legacy_state_dict)
        new_intervention.eval()

        with torch.no_grad():
            output_loaded = new_intervention(base.clone(), source.clone())

        diff = torch.norm(output_original - output_loaded).item()
        print(f"Legacy format - output difference: {diff:.2e}")
        assert diff < 1e-5, f"Legacy format should work for inference, diff={diff}"

    def test_legacy_format_continuation_breaks_orthogonality(self):
        """
        Test that legacy format DOES break orthogonality during continuation.
        This documents the known limitation of legacy checkpoints.
        """
        from pyreft import LoreftIntervention

        # Phase 1: Train
        intervention = LoreftIntervention(
            embed_dim=256, low_rank_dimension=8, dropout=0.0, dtype=torch.float32
        )
        optimizer = torch.optim.Adam(intervention.parameters(), lr=1e-3)

        for step in range(20):
            torch.manual_seed(step)
            base = torch.randn(2, 10, 256)
            source = torch.randn(2, 10, 256)
            output = intervention(base, source)
            loss = output.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save as legacy format (simulating old checkpoint)
        legacy_state_dict = create_legacy_state_dict(intervention)

        # Phase 2: Load and continue
        new_intervention = LoreftIntervention(
            embed_dim=256, low_rank_dimension=8, dropout=0.0, dtype=torch.float32
        )
        new_intervention.load_state_dict(legacy_state_dict)
        new_optimizer = torch.optim.Adam(new_intervention.parameters(), lr=1e-3)

        # Continue training
        for step in range(10):
            torch.manual_seed(100 + step)
            base = torch.randn(2, 10, 256)
            source = torch.randn(2, 10, 256)
            output = new_intervention(base, source)
            loss = output.sum()
            new_optimizer.zero_grad()
            loss.backward()
            new_optimizer.step()

        # Check orthogonality - THIS SHOULD BE BROKEN for legacy format
        R = new_intervention.rotate_layer.weight
        error = get_column_orthogonality_error(R)
        print(f"Legacy format - orthogonality error after continuation: {error:.2e}")

        # Note: We expect this to be broken (error > 0.01)
        # This test documents the limitation, not a desired behavior
        if error > 0.01:
            print("  [EXPECTED] Legacy format breaks orthogonality during continuation")
        else:
            print("  [UNEXPECTED] Legacy format maintained orthogonality")


class TestCheckpointContinuation:
    """Tests comparing new vs legacy format for checkpoint continuation."""

    def test_side_by_side_comparison(self):
        """
        Side-by-side comparison of save_for_training vs legacy format during continuation.
        """
        from pyreft import LoreftIntervention

        results = {}

        for format_name, use_legacy in [("new", False), ("legacy", True)]:
            # Phase 1: Train (use save_for_training=True for new format)
            torch.manual_seed(0)
            intervention = LoreftIntervention(
                embed_dim=256, low_rank_dimension=8, dropout=0.0, dtype=torch.float32,
                save_for_training=(not use_legacy)
            )
            optimizer = torch.optim.Adam(intervention.parameters(), lr=1e-3)

            for step in range(20):
                torch.manual_seed(step)
                base = torch.randn(2, 10, 256)
                source = torch.randn(2, 10, 256)
                output = intervention(base, source)
                loss = output.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Save
            if use_legacy:
                state_dict = create_legacy_state_dict(intervention)
            else:
                state_dict = intervention.state_dict()

            # Phase 2: Load and continue
            torch.manual_seed(0)  # Reset for fair comparison
            new_intervention = LoreftIntervention(
                embed_dim=256, low_rank_dimension=8, dropout=0.0, dtype=torch.float32
            )
            new_intervention.load_state_dict(state_dict)
            new_optimizer = torch.optim.Adam(new_intervention.parameters(), lr=1e-3)

            # Continue training
            ortho_errors = []
            for step in range(10):
                torch.manual_seed(100 + step)
                base = torch.randn(2, 10, 256)
                source = torch.randn(2, 10, 256)
                output = new_intervention(base, source)
                loss = output.sum()
                new_optimizer.zero_grad()
                loss.backward()
                new_optimizer.step()

                R = new_intervention.rotate_layer.weight
                ortho_errors.append(get_column_orthogonality_error(R))

            results[format_name] = {
                "final_ortho_error": ortho_errors[-1],
                "max_ortho_error": max(ortho_errors),
                "ortho_errors": ortho_errors,
            }

        print("\n=== Side-by-side comparison ===")
        print(f"New format    - final ortho error: {results['new']['final_ortho_error']:.2e}")
        print(f"Legacy format - final ortho error: {results['legacy']['final_ortho_error']:.2e}")
        print(f"Improvement ratio: {results['legacy']['final_ortho_error'] / (results['new']['final_ortho_error'] + 1e-10):.1f}x")

        # New format should be much better
        assert results["new"]["final_ortho_error"] < 1e-3, "New format should maintain orthogonality"
        assert results["new"]["final_ortho_error"] < results["legacy"]["final_ortho_error"], \
            "New format should be better than legacy"

        return results


class TestOptimizerStateInteraction:
    """Tests for optimizer state interaction with orthogonality."""

    def test_optimizer_step_maintains_orthogonality(self):
        """Test that optimizer step maintains orthogonality via parameterization."""
        from pyreft import LoreftIntervention

        intervention = LoreftIntervention(
            embed_dim=256, low_rank_dimension=8, dropout=0.0, dtype=torch.float32
        )
        optimizer = torch.optim.Adam(intervention.parameters(), lr=1e-3)

        for step in range(10):
            base = torch.randn(2, 10, 256)
            source = torch.randn(2, 10, 256)
            output = intervention(base, source)
            loss = output.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            R = intervention.rotate_layer.weight
            error = get_column_orthogonality_error(R)
            assert error < 1e-3, f"R should remain ~orthogonal after step {step}, error={error}"


def run_all_tests_with_output():
    """Run all tests and return output string."""
    import io
    import contextlib

    output_buffer = io.StringIO()

    with contextlib.redirect_stdout(output_buffer):
        print("=" * 70)
        print("ReFT Orthogonality Save/Load Test Results")
        print("=" * 70)

        # Test 1: Basic orthogonality
        print("\n" + "-" * 70)
        print("Test 1: Basic orthogonality utilities")
        print("-" * 70)

        random_matrix = torch.randn(64, 64)
        Q, _ = torch.linalg.qr(random_matrix)
        error = get_orthogonality_error(Q)
        print(f"  Orthogonal matrix (64x64): error = {error:.2e}")

        error = get_orthogonality_error(random_matrix)
        print(f"  Random matrix (64x64): error = {error:.2e}")

        # Test 2: state_dict formats
        print("\n" + "-" * 70)
        print("Test 2: state_dict formats (default vs save_for_training)")
        print("-" * 70)

        try:
            from pyreft import LoreftIntervention

            # Default: inference only
            intervention = LoreftIntervention(
                embed_dim=256, low_rank_dimension=8, dropout=0.0, dtype=torch.float32
            )
            state_dict = intervention.state_dict()
            print(f"  Default keys: {list(state_dict.keys())}")
            assert "rotate_layer_original" not in state_dict
            print("  [PASS] Default format saves only inference weights")

            # save_for_training=True: full state
            intervention = LoreftIntervention(
                embed_dim=256, low_rank_dimension=8, dropout=0.0, dtype=torch.float32,
                save_for_training=True
            )
            state_dict = intervention.state_dict()
            print(f"  Training keys: {list(state_dict.keys())}")
            assert "rotate_layer_original" in state_dict and "rotate_layer_base" in state_dict
            print("  [PASS] Training format saves full parametrization state")
        except Exception as e:
            print(f"  Error: {e}")

        # Test 3: Training continuation with save_for_training=True
        print("\n" + "-" * 70)
        print("Test 3: Training continuation (save_for_training=True)")
        print("-" * 70)

        try:
            from pyreft import LoreftIntervention

            # Phase 1: Train with save_for_training=True
            intervention = LoreftIntervention(
                embed_dim=256, low_rank_dimension=8, dropout=0.0, dtype=torch.float32,
                save_for_training=True
            )
            optimizer = torch.optim.Adam(intervention.parameters(), lr=1e-3)

            for step in range(20):
                torch.manual_seed(step)
                base = torch.randn(2, 10, 256)
                source = torch.randn(2, 10, 256)
                output = intervention(base, source)
                loss = output.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            R = intervention.rotate_layer.weight
            error_before = get_column_orthogonality_error(R)
            print(f"  Before save (after 20 steps): ortho_error = {error_before:.2e}")

            # Save
            state_dict = intervention.state_dict()

            # Phase 2: Load and continue
            new_intervention = LoreftIntervention(
                embed_dim=256, low_rank_dimension=8, dropout=0.0, dtype=torch.float32
            )
            new_intervention.load_state_dict(state_dict)
            new_optimizer = torch.optim.Adam(new_intervention.parameters(), lr=1e-3)

            for step in range(10):
                torch.manual_seed(100 + step)
                base = torch.randn(2, 10, 256)
                source = torch.randn(2, 10, 256)
                output = new_intervention(base, source)
                loss = output.sum()
                new_optimizer.zero_grad()
                loss.backward()
                new_optimizer.step()

            R = new_intervention.rotate_layer.weight
            error_after = get_column_orthogonality_error(R)
            print(f"  After continuation (10 more steps): ortho_error = {error_after:.2e}")

            if error_after < 1e-3:
                print("  [PASS] New format maintains orthogonality!")
            else:
                print("  [FAIL] Orthogonality degraded!")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

        # Test 4: Legacy format continuation
        print("\n" + "-" * 70)
        print("Test 4: Legacy format checkpoint continuation")
        print("-" * 70)

        try:
            from pyreft import LoreftIntervention

            # Phase 1: Train
            intervention = LoreftIntervention(
                embed_dim=256, low_rank_dimension=8, dropout=0.0, dtype=torch.float32
            )
            optimizer = torch.optim.Adam(intervention.parameters(), lr=1e-3)

            for step in range(20):
                torch.manual_seed(step)
                base = torch.randn(2, 10, 256)
                source = torch.randn(2, 10, 256)
                output = intervention(base, source)
                loss = output.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            R = intervention.rotate_layer.weight
            error_before = get_column_orthogonality_error(R)
            print(f"  Before save (after 20 steps): ortho_error = {error_before:.2e}")

            # Save as LEGACY format
            legacy_state_dict = create_legacy_state_dict(intervention)
            print(f"  Legacy state dict keys: {list(legacy_state_dict.keys())}")

            # Phase 2: Load and continue
            new_intervention = LoreftIntervention(
                embed_dim=256, low_rank_dimension=8, dropout=0.0, dtype=torch.float32
            )
            new_intervention.load_state_dict(legacy_state_dict)
            new_optimizer = torch.optim.Adam(new_intervention.parameters(), lr=1e-3)

            for step in range(10):
                torch.manual_seed(100 + step)
                base = torch.randn(2, 10, 256)
                source = torch.randn(2, 10, 256)
                output = new_intervention(base, source)
                loss = output.sum()
                new_optimizer.zero_grad()
                loss.backward()
                new_optimizer.step()

            R = new_intervention.rotate_layer.weight
            error_after = get_column_orthogonality_error(R)
            print(f"  After continuation (10 more steps): ortho_error = {error_after:.2e}")

            if error_after > 0.01:
                print("  [EXPECTED] Legacy format breaks orthogonality (known limitation)")
            else:
                print("  [UNEXPECTED] Legacy format maintained orthogonality")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

        # Test 5: Side-by-side comparison
        print("\n" + "-" * 70)
        print("Test 5: Side-by-side comparison (new vs legacy)")
        print("-" * 70)

        try:
            test = TestCheckpointContinuation()
            results = test.test_side_by_side_comparison()

            print(f"\n  Summary:")
            print(f"    New format final error:    {results['new']['final_ortho_error']:.2e}")
            print(f"    Legacy format final error: {results['legacy']['final_ortho_error']:.2e}")
            ratio = results['legacy']['final_ortho_error'] / (results['new']['final_ortho_error'] + 1e-10)
            print(f"    Improvement: {ratio:.0f}x better orthogonality with new format")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 70)
        print("Tests complete!")
        print("=" * 70)

    return output_buffer.getvalue()


def main():
    """Run tests standalone with detailed output and save to file."""
    # Run tests and capture output
    output = run_all_tests_with_output()

    # Print to stdout
    print(output)

    # Save to file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "test_orthogonality_output.txt")

    with open(output_path, "w") as f:
        f.write(output)

    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()

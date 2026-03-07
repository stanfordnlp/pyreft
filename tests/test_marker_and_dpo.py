import pytest

pytest.importorskip("pyvene")
pytest.importorskip("transformers")

from pyreft.dataset import (
    _parse_marker_spans,
    _span_ranges_to_token_indices,
    make_marker_supervised_data_module,
    make_dpo_data_module,
    ReftDPODataCollator,
)


class TestParseMarkerSpans:
    def test_single_span(self):
        text = "Hello <<reft>> world </reft>> here"
        cleaned, spans = _parse_marker_spans(text)
        assert cleaned == "Hello world here"
        assert spans == [(6, 11)]

    def test_no_marker(self):
        text = "Hello world"
        cleaned, spans = _parse_marker_spans(text)
        assert cleaned == "Hello world"
        assert spans == []

    def test_two_spans(self):
        text = "A <<reft>> one </reft>> B <<reft>> two </reft>> C"
        cleaned, spans = _parse_marker_spans(text)
        assert cleaned == "A one B two C"
        assert spans == [(2, 5), (8, 11)]

    def test_custom_markers(self):
        text = "x [s] foo [/s] y"
        cleaned, spans = _parse_marker_spans(text, start_marker="[s]", end_marker="[/s]")
        assert cleaned == "x foo y"
        assert spans == [(2, 5)]


class TestMarkerSupervisedDataModule:
    @pytest.fixture
    def tokenizer_and_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(name)
        return tokenizer, model

    def test_marker_data_module_structure(self, tokenizer_and_model):
        tokenizer, model = tokenizer_and_model
        inputs = ["Say <<reft>> hello </reft>>.", "No markers here."]
        outputs = ["Hi", "Bye"]
        out = make_marker_supervised_data_module(
            tokenizer, model, inputs, outputs, num_interventions=1
        )
        assert "train_dataset" in out
        assert "data_collator" in out
        ds = out["train_dataset"]
        assert len(ds) == 2
        assert "input_ids" in ds.column_names
        assert "intervention_locations" in ds.column_names
        assert "labels" in ds.column_names

    def test_marker_intervention_locations_in_bounds(self, tokenizer_and_model):
        tokenizer, model = tokenizer_and_model
        inputs = ["<<reft>> hello </reft>>"]
        outputs = [" world"]
        out = make_marker_supervised_data_module(
            tokenizer, model, inputs, outputs, num_interventions=1
        )
        row = out["train_dataset"][0]
        seq_len = len(row["input_ids"])
        for loc_list in row["intervention_locations"]:
            for idx in loc_list:
                assert 0 <= idx < seq_len or idx == -1


class TestSpanRangesToTokenIndices:
    def test_basic(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        text = "Hello world"
        spans = [(0, 5)]
        indices = _span_ranges_to_token_indices(tokenizer, text, spans)
        assert len(indices) == 1
        assert len(indices[0]) >= 1
        assert all(0 <= i for i in indices[0])


class TestDPODataModule:
    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained("gpt2")
        t.pad_token = t.eos_token
        return t

    def test_make_dpo_data_module_structure(self, tokenizer):
        prompts = ["Q: 1+1? A:", "Q: 2+2? A:"]
        chosen = [" 2", " 4"]
        rejected = [" 3", " 5"]
        out = make_dpo_data_module(
            tokenizer, prompts, chosen, rejected,
            max_length=64, num_interventions=1
        )
        assert "train_dataset" in out
        assert "data_collator" in out
        ds = out["train_dataset"]
        assert len(ds) == 2
        assert "prompt" in ds.column_names
        assert "chosen" in ds.column_names
        assert "rejected" in ds.column_names

    def test_dpo_collator_output_shape(self, tokenizer):
        collator = ReftDPODataCollator(
            tokenizer=tokenizer,
            max_length=32,
            num_interventions=1,
            pad_token_id=tokenizer.pad_token_id,
        )
        features = [
            {"prompt": "Q: x? A:", "chosen": " yes", "rejected": " no"},
            {"prompt": "Q: y? A:", "chosen": " no", "rejected": " yes"},
        ]
        batch = collator(features)
        assert batch["chosen_input_ids"].shape[0] == 2
        assert batch["rejected_input_ids"].shape[0] == 2
        assert batch["intervention_locations"].shape[0] == 2
        assert "chosen_labels" in batch
        assert "rejected_labels" in batch

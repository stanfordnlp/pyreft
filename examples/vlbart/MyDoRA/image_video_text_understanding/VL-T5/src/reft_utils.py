IGNORE_INDEX = -100
from transformers import DataCollatorForSeq2Seq
import torch

def parse_positions(positions: str):
    # parse position
    first_n, last_n = 0, 0
    if "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
    else:
        if "f" in positions:
            first_n = int(positions.strip("f"))
        elif "l" in positions:
            last_n = int(positions.strip("l"))
    return first_n, last_n


def get_intervention_locations(**kwargs):
    """
    This function generates the intervention locations.

    For your customized dataset, you want to create your own function.
    """
    # parse kwargs
    share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
    last_position = kwargs["last_position"]
    if "positions" in kwargs:
        _first_n, _last_n = parse_positions(kwargs["positions"])
    else:
        _first_n, _last_n = kwargs["first_n"], kwargs["last_n"]
    num_interventions = kwargs["num_interventions"]
    pad_mode = kwargs["pad_mode"] if "pad_mode" in kwargs else "first"
    last_offset = kwargs["last_offset"] if "last_offset" in kwargs else 0
    last_position += last_offset


    first_n = min(last_position // 2, _first_n)
    last_n = min(last_position // 2, _last_n)

    pad_amount = (_first_n - first_n) + (_last_n - last_n)
    pad_position = -1 if pad_mode == "first" else last_position
    if share_weights or (first_n == 0 or last_n == 0):
        position_list = [i for i in range(first_n)] + \
            [i for i in range(last_position - last_n, last_position)] + \
            [pad_position for _ in range(pad_amount)]
        intervention_locations = [position_list]*num_interventions
    else:
        left_pad_amount = (_first_n - first_n)
        right_pad_amount = (_last_n - last_n)
        left_intervention_locations = [i for i in range(first_n)] + [pad_position for _ in range(left_pad_amount)]
        right_intervention_locations = [i for i in range(last_position - last_n, last_position)] + \
            [pad_position for _ in range(right_pad_amount)]
        # after padding, there could be still length diff, we need to do another check
        left_len = len(left_intervention_locations)
        right_len = len(right_intervention_locations)
        if left_len > right_len:
            right_intervention_locations += [pad_position for _ in range(left_len-right_len)]
        else:
            left_intervention_locations += [pad_position for _ in range(right_len-left_len)]
        intervention_locations = [left_intervention_locations]*(num_interventions//2) + \
            [right_intervention_locations]*(num_interventions//2)
    
    return intervention_locations


def get_image_intervention_locations(**kwargs):
    """
    This function generates the intervention locations.
    For simplicity, this function does not implement padding.

    For your customized dataset, you want to create your own function.
    """
    # parse kwargs
    share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
    last_text_position = kwargs["last_position"]
    assert "image_positions" in kwargs, "Image positions must be provided"
    assert "positions" in kwargs, "Text positions must be provided"
    first_n, last_n = parse_positions(kwargs["positions"])
    first_image_n, last_image_n = parse_positions(kwargs["image_positions"])

    num_interventions = kwargs["num_interventions"]
    image_offset = kwargs["last_offset"] if "last_offset" in kwargs else 0

    pad_mode = kwargs["pad_mode"] if "pad_mode" in kwargs else "first"
    pad_position = -1 if pad_mode == "first" else last_text_position + image_offset
    if pad_mode != "first" and "nlvr" in kwargs["tasks"]:
        pad_position = last_text_position + 2 * image_offset

    if share_weights or ((first_n == 0 or last_n == 0) and (first_image_n == 0 or last_image_n == 0)):
        position_list = [i for i in range(first_n)] + \
            [i for i in range(last_text_position - last_n, last_text_position)]
        image_position_list = [i for i in range(last_text_position, last_text_position + first_image_n)] + \
            [i for i in range(last_text_position + image_offset - last_image_n, last_text_position + image_offset)]
        if "nlvr" in kwargs["tasks"]:
            image_position_list += [i for i in range(last_text_position + image_offset, last_text_position + image_offset + first_image_n)] + \
            [i for i in range(last_text_position + 2 * image_offset - last_image_n, last_text_position + 2 * image_offset)]
        text_len = len(position_list)
        image_len = len(image_position_list)
        if text_len > image_len:
            image_position_list += [pad_position for _ in range(text_len-image_len)]
        else:
            position_list += [pad_position for _ in range(image_len-text_len)]
        intervention_locations = [position_list]*(num_interventions//2) + \
            [image_position_list]*(num_interventions//2)
    else:
        assert first_n == last_n, "For now, we only support same first and last positions"
        left_intervention_locations = [i for i in range(first_n)]
        right_intervention_locations = [i for i in range(last_text_position - last_n, last_text_position)]
        left_image_intervention_locations = [i for i in range(last_text_position, last_text_position + first_image_n)]
        right_image_intervention_locations = [i for i in range(last_text_position + image_offset - last_image_n, last_text_position + image_offset)]
        if "nlvr" in kwargs["tasks"]:
            left_image_intervention_locations += [i for i in range(last_text_position + image_offset, last_text_position + image_offset + first_image_n)]
            right_image_intervention_locations += [i for i in range(last_text_position + 2 * image_offset - last_image_n, last_text_position + 2 * image_offset)]
        text_len = len(left_intervention_locations)
        image_len = len(left_image_intervention_locations)
        if text_len > image_len:
            left_image_intervention_locations += [pad_position for _ in range(text_len-image_len)]
            right_image_intervention_locations += [pad_position for _ in range(text_len-image_len)]
        else:
            left_intervention_locations += [pad_position for _ in range(image_len-text_len)]
            right_intervention_locations += [pad_position for _ in range(image_len-text_len)]

        intervention_locations = [left_intervention_locations]*(num_interventions//4) + \
            [right_intervention_locations]*(num_interventions//4) + \
            [left_image_intervention_locations]*(num_interventions//4) + \
            [right_image_intervention_locations]*(num_interventions//4)
    return intervention_locations

    
def compute_intervention(
    id: int, 
    result: dict, 
    tokenizer,
    fields_to_pad = [],
    fields_to_mask = [],
    **kwargs):
    pad_mode = kwargs["pad_mode"]
    # compute intervention locs
    if "image_positions" in kwargs:
        intervention_locations = get_image_intervention_locations(**kwargs)
    else:
        intervention_locations = get_intervention_locations(**kwargs)
    result["intervention_locations"] = intervention_locations
    result["id"] = id

    # add a single padding token BEFORE input_ids and fix everything
    if fields_to_pad is not None:
        if pad_mode == "first":
            for field in fields_to_pad:
                if field not in result:
                    continue
                if field == "labels":
                    result[field] = torch.cat((torch.tensor([IGNORE_INDEX,]), result[field]))
                else:
                    result[field] = torch.cat((torch.tensor([tokenizer.pad_token_id,]), result[field]))
            result["intervention_locations"] = (torch.IntTensor(result["intervention_locations"]) + 1).tolist()
            result["input_length"] += 1
        elif pad_mode == "last":
            for field in fields_to_pad:
                if field not in result:
                    continue
                if field == "labels":
                    result[field] = torch.cat((result[field], torch.tensor([IGNORE_INDEX,])))
                else:
                    result[field] = torch.cat((result[field], torch.tensor([tokenizer.pad_token_id,])))
            result["input_length"] += 1
        
    # attention masks
    if len(fields_to_mask) == 1:
        result["attention_mask"] = (result[fields_to_mask[0]] != tokenizer.pad_token_id).int()
    else:
        for field in fields_to_mask:
            result[f"{field}_mask"] = (result[field] != tokenizer.pad_token_id).int()

    # does not handle subspaces for now
    # print("Intervention Locations", result["intervention_locations"])
    return result

def reft_post_process(
    out_dict,
    tokenizer,
    idx: int, 
    last_position: int, 
    args = None,
    pad_mode = "none",
    fields_to_pad = [],
    fields_to_mask = []
):
    # print("Out_dict keys:", out_dict.keys())
    out_dict["instruction"] = tokenizer.decode(
        out_dict["input_ids"], skip_special_tokens=True)
    # out_dict["logits"] = out_dict["labels"]
    # out_dict["labels"] = out_dict["target_ids"]
    kwargs = {}
    if args is not None:
        kwargs["positions"] = args.positions
        kwargs["share_weights"] = args.share_weights
        layers = [int(l) for l in args.layers.split(";")]
        kwargs["num_interventions"] = len(layers) if args.share_weights else 2 * len(layers)
        if args.reft_image_rank != -1:
            kwargs["num_interventions"] *= 2
            kwargs["image_positions"] = args.image_positions
        kwargs["last_offset"] = args.n_boxes
        kwargs["pad_mode"] = pad_mode
        kwargs["last_position"] = last_position
        kwargs["tasks"] = args.prompt
    # print(kwargs)

    # print("BEFORE:", out_dict["input_ids"].shape, kwargs["last_position"])
    tokenized = compute_intervention(
            idx, 
            out_dict, 
            tokenizer,
            fields_to_pad,
            fields_to_mask,
            **kwargs)
    # print("AFTER:", tokenized["input_ids"].shape, tokenized["intervention_locations"])
    return tokenized

def keep_intervention_locations(datum):
    new_data = {}
    new_data["input_ids"] = datum["input_ids"]
    new_data["intervention_locations"] = datum["intervention_locations"]
    new_data["attention_mask"] = datum["attention_mask"]
    return new_data


def reft_supplemental_data_collator(batch, tokenizer):
    intervene_batch = [keep_intervention_locations(item) for item in batch]
    intervention_loc_collate_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        label_pad_token_id=-100,
        padding="longest"
    )
    intervene_batch_entry = intervention_loc_collate_fn(intervene_batch)

    batch_entry = {}
    id = []
    instructions = []
    for i, entry in enumerate(batch):
        if 'instruction' in entry:
            instructions.append(entry['instruction'])
        if 'id' in entry:
            id.append(entry['id'])
    import numpy as np
    batch_entry['id'] = np.array(id)
    batch_entry['instruction'] = instructions
    
    
    if "intervention_locations" in batch[0]:
        # print("Locs before data collator",intervene_batch_entry["input_ids"].shape, intervene_batch_entry["intervention_locations"].shape, intervene_batch_entry["intervention_locations"])
        batch_entry["intervention_locations"] = intervene_batch_entry["intervention_locations"]
        # print("Locs after data collator",batch_entry["intervention_locations"])
    return batch_entry

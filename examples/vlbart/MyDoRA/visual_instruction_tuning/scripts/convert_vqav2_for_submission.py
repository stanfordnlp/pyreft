import os
import argparse
import json

from llava.eval.m4c_evaluator import EvalAIAnswerProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./playground/data/eval/vqav2")
    parser.add_argument('--eval_dir', type=str, default="./playground/data/eval/vqav2")
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    print(f'ckpt: {args.ckpt}')
    src = os.path.join(args.save_dir, 'answers', args.split, args.ckpt, 'merge.jsonl')
    test_split = os.path.join(args.eval_dir, 'llava_vqav2_mscoco_test2015.jsonl')
    dst = os.path.join(args.save_dir, 'answers_upload', args.split, f'{args.ckpt}.json')
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    results = []
    error_line = 0
    for line_idx, line in enumerate(open(src)):
        try:
            line_ = json.loads(line)
            print(line_)
            results.append(line_)
        except:
            error_line += 1

    results = {x['question_id']: x['text'] for x in results}
    test_split = [json.loads(line) for line in open(test_split)]
    split_ids = set([x['question_id'] for x in test_split])

    print(f'total results: {len(results)}, total split: {len(test_split)}, error_line: {error_line}')

    all_answers = []

    answer_processor = EvalAIAnswerProcessor()

    for x in test_split:
        if x['question_id'] not in results:
            all_answers.append({
                'question_id': x['question_id'],
                'answer': ''
            })
        else:
            all_answers.append({
                'question_id': x['question_id'],
                'answer': answer_processor(results[x['question_id']])
            })

    with open(dst, 'w') as f:
        json.dump(all_answers, open(dst, 'w'))

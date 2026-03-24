# eval/aggregate/

Drop team JSONL result files here before running the aggregate script.

## File naming convention

```
{name}_eval_results_{eval_type}.jsonl
```

### Examples
```
vinod_eval_results_h2_global.jsonl      ← Eval 1, H2, Global model, all languages
vinod_eval_results_h1_fire_hi.jsonl     ← Eval 1, H1, Fire model, Hindi only
vinod_eval_results_h1_global_hi.jsonl   ← Eval 1, H1, Global model, Hindi only
vinod_eval_results_eval2_global.jsonl   ← Eval 2, LLM QA, Global model

abhishek_eval_results_h2_global.jsonl
randy_eval_results_eval2_global.jsonl
wahyu_eval_results_h2_global.jsonl
```

## Rules

- Always prefix your name — e.g. `abhishek_`, `randy_`, `wahyu_`, `paarth_`
- Use `results` not `report` in the filename
- Only `.jsonl` files — not the `.txt` report files
- The aggregate script deduplicates on (doc_id, question, model, field)
  so submitting the same file twice will not inflate the numbers

## Running the aggregate

```bash
# Dry run — see numbers without changing anything
python -m eval.aggregate --dry-run

# Full run — updates dashboard + saves eval/results/eval_results_aggregate.jsonl
python -m eval.aggregate
```

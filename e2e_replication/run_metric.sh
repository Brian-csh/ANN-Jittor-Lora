python LoRA/examples/NLG/src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/e2e/predict.35000.b10p08r4.jsonl \
    --input_file LoRA/examples/NLG/data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref.txt \
    --output_pred_file e2e_pred.txt

python LoRA/examples/NLG/eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p
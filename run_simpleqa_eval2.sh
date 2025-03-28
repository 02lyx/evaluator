#!/bin/bash
wget -qO- https://astral.sh/uv/install.sh | sh
uv venv briter --python 3.11 && source briter/bin/activate && uv pip install --upgrade pip --link-mode=copy
uv pip install -r requirements.txt --link-mode=copy
uv pip install flash_attn --no-build-isolation --link-mode=copy
uv uninstall wandb
uv pip install wandb --no-cache-dir --link-mode=copy
# Parameter settings
GENERATION_MODEL="Yuanxin-Liu/Qwen2.5-7B_Mix-Math-yt-rbt-grpo_0.5_exp12_gen_8_test_8_clip_ratio_-1_outer_kl-320"  # Model for generating answers
EVALUATION_MODEL="Qwen/QwQ-32B"  # Model for evaluating answers
ALL_GPUS="0,1,2,3"               # Available GPU IDs
ANSWERS_FILE="simpleqa_answers_without_clip_320_${DATASET_SIZE}_${NUM_SAMPLES}.jsonl"  # File to store generated answers
RESULTS_FILE="simpleqa_results_without_clip_320_${DATASET_SIZE}_${NUM_SAMPLES}.json"   # File to store evaluation results
BATCH_SIZE=8                     # Batch size
TEMPERATURE=1                    # Generation temperature
NUM_SAMPLES=8                    # Number of samples per question
DATASET_SIZE=100                 # Dataset size, -1 to use all
DATASET_NAME="simpleqa"
MAX_TOKENS=8192                  # Maximum tokens to generate

echo "=== SimpleQA Evaluation Pipeline ==="
echo "Generation model: $GENERATION_MODEL"
echo "Evaluation model: $EVALUATION_MODEL"
echo "Available GPUs: $ALL_GPUS"
echo "Temperature: $TEMPERATURE"
echo "Samples per question: $NUM_SAMPLES"
echo "Dataset size: $DATASET_SIZE"
echo "Dataset name: $DATASET_NAME"
echo "Maximum tokens to generate: $MAX_TOKENS"

# Step 1: Generate answers using the model
echo -e "\n=== Step 1: Generating answers using $GENERATION_MODEL ==="
python -m generate_answers \
    --model $GENERATION_MODEL \
    --output_file $ANSWERS_FILE \
    --gpu_ids $ALL_GPUS \
    --batch_size $BATCH_SIZE \
    --temperature $TEMPERATURE \
    --num_samples $NUM_SAMPLES \
    --dataset_size $DATASET_SIZE \
    --dataset_name $DATASET_NAME \
    --max_tokens $MAX_TOKENS

# Check if generation was successful
if [ ! -f "$ANSWERS_FILE" ]; then
    echo "Error: Failed to generate answers. File $ANSWERS_FILE not found."
    exit 1
fi

# Step 2: Evaluate answers using the model
echo -e "\n=== Step 2: Evaluating answers using $EVALUATION_MODEL ==="
python -m evaluate_answers \
    --model $EVALUATION_MODEL \
    --input_file $ANSWERS_FILE \
    --output_file $RESULTS_FILE \
    --gpu_ids $ALL_GPUS \
    --batch_size $BATCH_SIZE

# Check if evaluation was successful
if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: Failed to evaluate answers. File $RESULTS_FILE not found."
    exit 1
fi

echo -e "\n=== Evaluation completed successfully ==="
echo "Generated answers saved to: $ANSWERS_FILE"
echo "Evaluation results saved to: $RESULTS_FILE"

# Display evaluation results summary
echo -e "\n=== Summary of evaluation results ==="
python -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)
metrics = data['metrics']
print(f\"Total questions: {metrics['total_questions']}  \")
print(f\"Total samples: {metrics['total_samples']} ({metrics['samples_per_question']:.1f} samples/question)\")
print(f\"Correct: {metrics['correct_count']} ({metrics['correct_count']/metrics['total_samples']:.2%})\")
print(f\"Incorrect: {metrics['incorrect_count']} ({metrics['incorrect_count']/metrics['total_samples']:.2%})\")
print(f\"Not attempted: {metrics['not_attempted_count']} ({metrics['not_attempted_count']/metrics['total_samples']:.2%})\")
print(f\"Overall accuracy: {metrics['accuracy']:.2%}\")
print(f\"Accuracy given attempted: {metrics['accuracy_given_attempted']:.2%}\")
print(f\"Average question accuracy: {metrics['avg_question_accuracy']:.2%}\")
print(f\"Average question accuracy given attempted: {metrics['avg_question_accuracy_given_attempted']:.2%}\")
print(f\"F1 Score: {metrics['f1']:.3f}\")

# Print examples of best performing questions
if len(data['per_question_metrics']) > 0:
    print(f\"\nBest performing questions (accuracy=1.0):\")
    best_questions = [q for q in data['per_question_metrics'] if q['accuracy'] == 1.0][:3]
    for i, q in enumerate(best_questions):
        print(f\"Question {i+1}: {q['question']}\")
        print(f\"Answer: {q['target']}\")
        print(f\"Accuracy: {q['correct_count']}/{q['total_samples']}\")
        print()

    print(f\"Worst performing questions (accuracy=0.0):\")
    worst_questions = [q for q in data['per_question_metrics'] if q['accuracy'] == 0.0][:3]
    for i, q in enumerate(worst_questions):
        print(f\"Question {i+1}: {q['question']}\")
        print(f\"Answer: {q['target']}\")
        print(f\"Accuracy: {q['correct_count']}/{q['total_samples']}\")
        print()
" 
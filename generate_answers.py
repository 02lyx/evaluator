import os
import pandas as pd
import json
import argparse
import random
from vllm import LLM, SamplingParams
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Generate answers for SimpleQA dataset using a specified model')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B", help='Model name or path')
    parser.add_argument('--output_file', type=str, default="answers.jsonl", help='Output file path')
    parser.add_argument('--gpu_ids', type=str, default="0,1", help='Comma-separated GPU IDs to use')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--max_tokens', type=int, default=8192, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.1, help='Sampling temperature')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate per prompt')
    parser.add_argument('--dataset_size', type=int, default=-1, help='Number of examples to use from dataset, -1 to use all')
    parser.add_argument('--dataset_name', type=str, default="simpleqa", help='Dataset name')
    args = parser.parse_args()

    # Set visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    num_gpus = len(args.gpu_ids.split(','))

    if args.dataset_name == "simpleqa":
        print(f"Loading dataset from https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv")
        df = pd.read_csv("https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv")
        
        # If dataset size is specified, randomly select samples
        if args.dataset_size > 0 and args.dataset_size < len(df):
            print(f"Randomly selecting {args.dataset_size} examples from the dataset")
            df = df.sample(args.dataset_size, random_state=42)
        
        questions = df["problem"].tolist()
        targets = df["answer"].tolist()
    else:
        raise ValueError(f"Dataset name {args.dataset_name} not supported")
    
    print(f"Using {len(questions)} questions from dataset")
    print(f"Generating {args.num_samples} samples per question with temperature {args.temperature}")
    print(f"Maximum generation length set to {args.max_tokens} tokens")

    print(f"Loading model {args.model} on {num_gpus} GPUs: {args.gpu_ids}")
    
    # Initialize LLM instance
    llm = LLM(
        model=args.model,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=0.9,
        max_model_len=8192
    )

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1,
        max_tokens=args.max_tokens,
        n=args.num_samples
    )

    # Create list for all results
    all_results = []
    
    # Process questions in batches
    for i in tqdm(range(0, len(questions), args.batch_size)):
        batch_questions = questions[i:i+args.batch_size]
        batch_targets = targets[i:i+args.batch_size]
        batch_indices = list(range(i, min(i+args.batch_size, len(questions))))
        
        # Use llm.generate to process batch
        outputs = llm.generate(batch_questions, sampling_params)
        
        # Process results
        for j, output in enumerate(outputs):
            idx = batch_indices[j]
            for sample_idx, generated_text in enumerate(output.outputs):
                all_results.append({
                    "question_id": idx,
                    "sample_id": sample_idx,
                    "question": batch_questions[j],
                    "target": batch_targets[j],
                    "predicted_answer": generated_text.text
                })
    
    # Save results to file
    print(f"Saving {len(all_results)} generated answers to {args.output_file}")
    with open(args.output_file, "w") as f:
        for item in all_results:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main() 
import pathlib
import argparse
import torch

import numpy as np

from cs336_basics.data_loader import data_loader
from cs336_basics.optimizer import AdamW, gradient_clipping, cosine_lr_scheduler
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import cross_entropy_loss, softmax, transformer_lm
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint

"""
1. Instantiate Model and Optimizer with given hyperparameters
2. Get batches of data with np.memmap
3. Do a forward pass, calculate the loss, do the backward pass, 
log the results, make the optimizer step, and repeat.
4. When done with the iterations in step 3, log the metrics, 
and save the checkpoint.
"""

VOCAB_PATH = "cs336_basics/output/tiny_stories_vocab.pkl"
MERGES_PATH = "cs336_basics/output/tiny_stories_merges.pkl"

parser = argparse.ArgumentParser()

parser.add_argument("--context_length", type=int, required=True)
parser.add_argument("--num_layers", type=int, required=True)
parser.add_argument("--num_heads", type=int, required=True)
parser.add_argument("--d_model", type=int, required=True)
parser.add_argument("--d_ff", type=int, required=True)
parser.add_argument("--iterations", type=int, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--learning_rate", type=float, required=True)
parser.add_argument("--eval_steps", type=int, required=True)
parser.add_argument("--checkpoint_steps", type=int, required=True)


def train(model, optim, device, args):
    train_path = pathlib.Path("data/tiny_stories_train_tokens.npy")
    train_data = np.load(train_path, mmap_mode="r")

    val_path = pathlib.Path("data/tiny_stories_valid_tokens.npy")
    val_data = np.load(val_path, mmap_mode="r")

    for i in range(args.iterations):
        inputs, targets = data_loader(
            train_data, batch_size=args.batch_size,
            context_length=args.context_length, device=device
        )
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        if i % args.eval_steps == 0:
            print(f"Train loss at iteration {i}: {loss.item():.4f}")

        loss.backward()
        optim.step()
        optim.zero_grad()

        if i % args.eval_steps == 0:
            # Implement averaging more than one validation batches to reduce the noise
            with torch.no_grad():
                inputs, targets = data_loader(
                    val_data, batch_size=args.batch_size,
                    context_length=args.context_length, device=device
                )
                logits = model(inputs)
                loss = cross_entropy_loss(logits, targets)

                print(f"Valid loss at iteration {i}: {loss.item():.4f}")

        if i % args.checkpoint_steps == 0 and i != 0:
            save_checkpoint(model, optim, iteration=i, out=f"checkpoints/iteration_{i}.pt")

    torch.save(model.state_dict(), "checkpoints/final_model.pt")

def generate(
    tokenizer,
    model: torch.nn.Module,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    device
):

    # 1) Encode the tokens
    encoded = tokenizer.encode(prompt)
    
    # 2) Add batch dimension
    inputs = torch.Tensor(encoded).to(dtype=torch.int32, device=device).unsqueeze(0)
    
    for _ in range(max_output_tokens):    
        
        # 3) Input them to model for getting the logits
        logits = model(inputs)

        # 4) Apply temperature scaling and softmax
        logits /= temperature
        probs = softmax(logits, dim=-1)

        # 5) Apply top-p to get the truncated and apply softmax again
        values, indices = torch.sort(probs[0, -1], dim=-1, descending=True)
        prob_acc = 0.0
        i = 0
        while prob_acc < top_p:
            prob_acc += values[i]
            i += 1
        keep_indices = indices[:i]

        dont_keep_mask = torch.ones_like(probs[0, -1], dtype=torch.bool)
        dont_keep_mask[keep_indices] = False
        probs[0, -1, dont_keep_mask] = 0
        probs[0, -1, :] /= torch.sum(probs[0, -1, :])
        probs = softmax(probs, dim=-1)

        # 6) Sample one token from that distribution
        selected_token = torch.multinomial(probs[0, -1], num_samples=1).unsqueeze(0)
        
        # 7) Append it to the end of the sequence
        inputs = torch.cat([inputs, selected_token], dim=-1)
        
        if tokenizer.decode([selected_token.item()]) == "<|endoftext|>":
            break
    print(tokenizer.decode(inputs.squeeze(dim=0).tolist()))


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("mps")
    
    # Instantiate the tokenizer from saved vocab and merges
    tokenizer = Tokenizer.from_files(
        VOCAB_PATH,
        MERGES_PATH,
        special_tokens=["<|endoftext|>"]
    )

    model = transformer_lm(
        vocab_size=len(tokenizer.vocab),
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        device=device,
        dtype=torch.float32
    )

    optim = AdamW(
        params=model.parameters(),
        lr=args.learning_rate,
    )
    generate(
        tokenizer, model, prompt="hello world",
        max_output_tokens=10, temperature=0.8, top_p=0.9, device=device
    )

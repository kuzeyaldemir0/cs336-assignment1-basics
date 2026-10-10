import pathlib
import argparse
import torch

import numpy as np
import wandb

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
"""

VOCAB_PATH = "cs336_basics/output/tiny_stories_vocab.pkl"
MERGES_PATH = "cs336_basics/output/tiny_stories_merges.pkl"

parser = argparse.ArgumentParser()

parser.add_argument("--num_layers", type=int, required=True)
parser.add_argument("--num_heads", type=int, required=True)
parser.add_argument("--d_model", type=int, required=True)
parser.add_argument("--d_ff", type=int, required=True)
parser.add_argument("--context_length", type=int, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--total_steps", type=int, required=True)
parser.add_argument("--warmup_percentage", type=float, default=0.01)
parser.add_argument("--init_lr", type=float, required=True)
parser.add_argument("--max_lr", type=float, required=True)
parser.add_argument("--min_lr", type=float, required=True)
parser.add_argument("--run_name", type=str, default=None)


def train(model, optim, device, args):
    model.train()
    train_path = pathlib.Path("data/tiny_stories_train_tokens.npy")
    train_data = np.load(train_path, mmap_mode="r")

    val_path = pathlib.Path("data/tiny_stories_valid_tokens.npy")
    val_data = np.load(val_path, mmap_mode="r")

    best_val_loss = float("inf")
    warmup_steps = args.total_steps * args.warmup_percentage
    eval_steps = 200
    eval_batch_count = round((
        len(val_data) / args.batch_size / args.context_length 
    ) * 0.1)

    for i in range(args.total_steps):
        for group in optim.param_groups:
            group["lr"] = cosine_lr_scheduler(
                it=i, 
                lr_max=args.max_lr,
                lr_min=args.min_lr,
                warmup_it=warmup_steps,
                cosine_cycle_it=args.total_steps
            )

        # Forward and backward pass
        inputs, targets = data_loader(
            train_data, batch_size=args.batch_size,
            context_length=args.context_length, device=device
        )
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        loss.backward()
        optim.step()
        optim.zero_grad()
        wandb.log({"train_loss": loss.item(), "lr": optim.param_groups[0]["lr"]}, step=i)

        # Evaluation with val loss averaged over multiple batches
        if i % eval_steps == 0 or i == args.total_steps - 1:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for _ in range(eval_batch_count):
                    inputs, targets = data_loader(
                        val_data, batch_size=args.batch_size,
                        context_length=args.context_length, device=device
                    )
                    logits = model(inputs)
                    loss = cross_entropy_loss(logits, targets)
                    val_loss += loss.item()
                val_loss /= eval_batch_count
                wandb.log({"val_loss": val_loss}, step=i)
                
                # Save the checkpoint with the best validation loss
                if val_loss <= best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f"{args.run_name}.pth")
            model.train()

def generate(
    tokenizer,
    model: torch.nn.Module,
    prompt: str,
    context_length: int,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    device
):
    assert temperature > 0.0, "Temperature should be a positive value."
    model.eval()
    with torch.no_grad():
        # 1) Encode the tokens
        encoded = tokenizer.encode(prompt)
        # 2) Add batch dimension
        inputs = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_output_tokens): 
            if inputs.shape[-1] >= context_length:
                print("Context is at maximum length.")
                break
            # 3) Input them to model for getting the logits
            logits = model(inputs)

            # 4) Apply temperature scaling and softmax
            logits /= temperature
            probs = softmax(logits, dim=-1)
            next_token_probs = probs[0, -1]

            # 5) Apply top-p to get the truncated probs and renormalize
            values, indices = torch.sort(next_token_probs, dim=-1, descending=True)
            prob_acc = 0.0
            i = 0
            while prob_acc < top_p and i < next_token_probs.shape[-1]:
                prob_acc += values[i]
                i += 1
            keep_indices = indices[:i]

            # Create a mask for the ones we truncate and make their probs 0
            dont_keep_mask = torch.ones_like(next_token_probs, dtype=torch.bool)
            dont_keep_mask[keep_indices] = False
            next_token_probs[dont_keep_mask] = 0
            
            # Renormalize so that that the probs sum to 1
            next_token_probs /= torch.sum(next_token_probs)

            # 6) Sample one token from that distribution
            selected_token = torch.multinomial(next_token_probs, num_samples=1).unsqueeze(0)
            if tokenizer.decode([selected_token.item()]) == "<|endoftext|>":
                break

            # 7) Append it to the end of the sequence
            inputs = torch.cat([inputs, selected_token], dim=-1)
    print(tokenizer.decode(inputs.squeeze(dim=0).tolist()))


if __name__ == "__main__":
    args = parser.parse_args()
    wandb.init(project="cs336-a1", config=vars(args), group="batch-size-sweep", name=args.run_name)
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
        lr=args.init_lr,
    )

    """
    state_dict = torch.load(
        "checkpoints/lr_sweep/4000-steps-increase-max-init-lr.pth",
        map_location=torch.device("mps")
    )
    model.load_state_dict(state_dict)
    generate(
        tokenizer, model, "Once upon a time, there was a little girl named Alice",
        context_length=args.context_length,
        max_output_tokens=300, temperature=0.8, top_p=0.8, device=device        
    )
    """
    
    train(model, optim, device, args)
    wandb.finish()

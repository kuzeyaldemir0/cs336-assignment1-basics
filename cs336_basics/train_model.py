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
parser.add_argument("--warmup_steps", type=int, default=None)
parser.add_argument("--init_lr", type=float, required=True)
parser.add_argument("--max_lr", type=float, required=True)
parser.add_argument("--min_lr", type=float, required=True)
parser.add_argument("--cosine_lr_scheduler", type=bool, default=True)
parser.add_argument("--run_name", type=str, default=None)


def train(model, optim, device, args):
    train_path = pathlib.Path("data/tiny_stories_train_tokens.npy")
    train_data = np.load(train_path, mmap_mode="r")

    val_path = pathlib.Path("data/tiny_stories_valid_tokens.npy")
    val_data = np.load(val_path, mmap_mode="r")

    eval_steps = 200
    eval_batch_count = round((
        len(val_data) / args.batch_size / args.context_length 
    ) * 0.1)

    if args.warmup_steps is None:
        args.warmup_steps = args.total_steps / 0.1

    for i in range(args.total_steps):
        inputs, targets = data_loader(
            train_data, batch_size=args.batch_size,
            context_length=args.context_length, device=device
        )
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        loss.backward()
        optim.step()
        optim.zero_grad()

        if args.cosine_lr_scheduler:
            for group in optim.param_groups:
                group["lr"] = cosine_lr_scheduler(
                    it=i, 
                    lr_max=args.max_lr,
                    lr_min=args.min_lr,
                    warmup_it=args.warmup_steps,
                    cosine_cycle_it=args.total_steps
                )

        wandb.log({"train_loss": loss.item()}, step=i)
        if i % eval_steps == 0:
            # Calculate val loss averaged over multiple batches
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

    torch.save(model.state_dict(), f"{args.run_name}.pth")

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
    train(model, optim, device, args)
    wandb.finish()

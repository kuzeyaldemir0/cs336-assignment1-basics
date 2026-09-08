import pickle
import time
from weakref import ref

import regex as re

from collections import defaultdict


def int_byte_vocab_init(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for idx, token in enumerate(special_tokens):
        vocab[256 + idx] = token.encode("utf-8")
    return vocab


def pretoken_str_to_tuple_of_bytes(
        pre_token_freqs: defaultdict[str, int]
) -> dict[tuple[bytes, ...], int]:
    bytes_freqs = {}
    for pre_token, occurence in pre_token_freqs.items():
        byte_data = pre_token.encode("utf-8")

        # Use a comprehension to split them into single bytes objects
        tuple_of_bytes = tuple(bytes([b]) for b in byte_data)
        bytes_freqs[tuple_of_bytes] = occurence
    return bytes_freqs


def successive_pair_freq(pretoken_freqs: defaultdict):
    pair_freqs = defaultdict(int)
    pair_pretoken_lookup = defaultdict(set)
    for pretoken, freq in pretoken_freqs.items():
        # Loop over each successive pair
        for i in range(len(pretoken) - 1):
            # Add the occurence of that word to the pair's frequency
            pair_freqs[(pretoken[i], pretoken[i+1])] += freq

            # Use the successive pairs as keys and append the tuple to the list as the value
            pair_pretoken_lookup[(pretoken[i], pretoken[i+1])].add(pretoken)
    return pair_freqs, pair_pretoken_lookup


def merge_pair(
        pretoken_freqs: defaultdict[tuple[bytes, ...], int],
        pair_freqs: defaultdict[tuple[bytes, bytes], int],
        pair_pretoken_lookup: defaultdict[tuple[bytes, bytes], set[tuple[bytes, ...]]], 
        pair: tuple[bytes, bytes]
):
    merged_pair = pair[0] + pair[1]
    matched_pretokens = pair_pretoken_lookup[pair].copy()
    # pretoken example for a word started with low: (b"lo", b"w")
    for pretoken in matched_pretokens:
        merged_pretoken = []
        i = 0
        if len(pretoken) == 1:
            merged_pretoken.append(pretoken[0])

        # Walk each bytes object inside the pretoken and
        # create the new pretoken that has our merged pair as one bytes object
        while i < len(pretoken) - 1:
            if (pretoken[i], pretoken[i+1]) != pair:
                # If it's not our pair, append the current bytes object to the new tuple
                merged_pretoken.append(pretoken[i])
            elif (pretoken[i], pretoken[i+1]) == pair:
                # Add our merged pair to the merged_pretoken that accumulates bytes objects one by one
                merged_pretoken.append(merged_pair)

                # Skip next bytes object since we merged it
                i += 1
            if i == len(pretoken) - 2:
                # Add the last element if still no merge in the last index
                merged_pretoken.append(pretoken[i+1])
            i += 1

        # Convert the list to a tuple for it to be a dictionary key
        merged_pretoken = tuple(merged_pretoken)

        i = 0
        while i < len(pretoken) - 1:
            pair_freqs[pretoken[i], pretoken[i+1]] -= pretoken_freqs[pretoken]
            pair_pretoken_lookup[pretoken[i], pretoken[i+1]].discard(pretoken)
            i += 1

        i = 0
        while i < len(merged_pretoken) - 1:
            pair_freqs[merged_pretoken[i], merged_pretoken[i+1]] += pretoken_freqs[pretoken]
            pair_pretoken_lookup[merged_pretoken[i], merged_pretoken[i+1]].add(merged_pretoken)
            i += 1            

        pretoken_freqs[merged_pretoken] = pretoken_freqs[pretoken]
        del pretoken_freqs[pretoken]

    del pair_freqs[pair]


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # Initialize vocabulary
    vocab = int_byte_vocab_init(special_tokens)

    # Read the validation set for faster debugging
    with open(input_path, "r", encoding="utf-8") as f:
        corpus = f.read()
        # Validation has 22,493,387 chars

    # Escape the '|' in special tokens
    escaped_special_tokens = []
    for sp_token in special_tokens:
        escaped_special_tokens.append(re.escape(sp_token))

    # Join all the special tokens with or ('|') operator for the regex pattern
    special_token_pattern = "|".join(escaped_special_tokens)

    # Split the corpus on each special token
    chunks = re.split(special_token_pattern, corpus)

    # Pre-tokenize and combine the frequencies of pre-tokens across chunks
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretoken_freqs = defaultdict(int)
    for chunk in chunks:
        # Pre-tokenize and add the frequency of pre-token
        for match in re.finditer(PAT, chunk):
            pretoken = match.group()
            pretoken_freqs[pretoken] += 1

    # Convert the str type pre-token object to tuple of bytes for each pre-token
    pretoken_freqs = pretoken_str_to_tuple_of_bytes(pretoken_freqs)
    pair_freqs, pair_pretoken_lookup = successive_pair_freq(pretoken_freqs)

    # Start merging
    merges = []
    num_merges = vocab_size - (len(special_tokens) + 256)
    for i in range(num_merges):
        # Select the most frequent successive pair across chunks
        # If tied, take the alphabetically greater/latter pair
        most_frequent_pair = max(pair_freqs, key=lambda pair: (pair_freqs[pair], pair))

        # Merge the most frequent pair across all the chunks and change stuff in-place
        merge_pair(pretoken_freqs, pair_freqs, pair_pretoken_lookup, most_frequent_pair)

        # Append the resulting merge to the merges list
        merges.append(most_frequent_pair)

        # Add the new merged token to the vocab
        vocab[256 + len(special_tokens) + i] = most_frequent_pair[0] + most_frequent_pair[1]

    return vocab, merges

if __name__ == "__main__":
    start_time = time.perf_counter()
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    elapsed_time = time.perf_counter() - start_time
    print(f"Total Duration: {elapsed_time:.2f} seconds")

    with open("cs336_basics/output/vocab.pkl", "wb") as file:
        pickle.dump(vocab, file)
    with open("cs336_basics/output/merges.pkl", "wb") as file:
        pickle.dump(merges, file)
                        
import pickle
import time

import regex as re

from collections import defaultdict

import cs336_basics


def int_byte_vocab_init(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for idx, token in enumerate(special_tokens):
        vocab[256 + idx] = token.encode("utf-8")
    return vocab


def pre_token_to_tuple_of_bytes(
        pre_token_freqs: defaultdict[str, int]
) -> dict[tuple[bytes, ...], int]:
    bytes_freqs = {}
    for pre_token, occurence in pre_token_freqs.items():
        byte_data = pre_token.encode("utf-8")

        # Use a comprehension to split them into single bytes objects
        tuple_of_bytes = tuple(bytes([b]) for b in byte_data)
        bytes_freqs[tuple_of_bytes] = occurence
    return bytes_freqs


def successive_pair_freq(bytes_freqs: defaultdict) -> defaultdict:
    pair_freqs = defaultdict(int)
    for bytes_tuple, occurence in bytes_freqs.items():
        # Loop over each successive pair
        for i in range(len(bytes_tuple) - 1):
            # Add the occurence of that word to the pair's frequency
            pair_freqs[(bytes_tuple[i], bytes_tuple[i+1])] += occurence
    return pair_freqs


def merge_pair(bytes_freqs, pair):
    new_byte_freqs = {}
    # Walk through each tuple of bytes, merge bytes objects if they're our pair
    for bytes_tuple, occurence in bytes_freqs.items():
        bytes_list = []
        i = 0
        if len(bytes_tuple) == 1:
            bytes_list.append(bytes_tuple[0])
        while i < len(bytes_tuple) - 1:

            # If current successive pair is not our pair,
            if (bytes_tuple[i], bytes_tuple[i+1]) != pair:

                # append current bytes object to our list
                bytes_list.append(bytes_tuple[i])

            # If current successive pair is the pair we want to merge,
            else:

                # append the merged bytes object to the list
                bytes_list.append(bytes_tuple[i] + bytes_tuple[i+1])

                # Skip the next bytes object since we merged the 2nd one
                i += 1

            # If we're on the last index we're going iterate
            if i == len(bytes_tuple) - 2:

                # Add last element to the list too
                bytes_list.append(bytes_tuple[i+1])

            # Iterate to the next successive pair
            i += 1

        # Convert the list to a tuple for it to be a dictionary key
        new_key = tuple(bytes_list)
        new_byte_freqs[new_key] = occurence
    return new_byte_freqs


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
    pre_token_freqs = defaultdict(int)
    for chunk in chunks:
        # Pre-tokenize and add the frequency of pre-token
        for match in re.finditer(PAT, chunk):
            pre_token = match.group()
            pre_token_freqs[pre_token] += 1

    # Convert the str type pre-token object to tuple of bytes for each pre-token
    bytes_freqs = pre_token_to_tuple_of_bytes(pre_token_freqs)

    # Start merging
    merges = []
    num_merges = vocab_size - (len(special_tokens) + 256)
    for i in range(num_merges):
        # Count successive pairs of bytes objects in each tuple of bytes across all chunks
        pair_freqs = successive_pair_freq(bytes_freqs)

        # Select the most frequent successive pair across chunks
        # If tied, take the alphabetically greater/latter pair
        most_frequent_pair = max(pair_freqs, key=lambda pair: (pair_freqs[pair], pair))

        # Merge the most frequent pair across all the chunks
        bytes_freqs = merge_pair(bytes_freqs, most_frequent_pair)

        # Append the resulting merge to the merges list
        merges.append(most_frequent_pair)

        # Add the new merged token to the vocab
        vocab[256 + len(special_tokens) + i] = most_frequent_pair[0] + most_frequent_pair[1]

    return vocab, merges

if __name__ == "__main__":
    start_time = time.perf_counter()
    vocab, merges = train_bpe(
        input_path="data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    elapsed_time = time.perf_counter() - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    # It ran for 65 seconds for the validation set.
    # Load and view the pickle files to check them
    # Do profiling to see how we can improve the speed with leverage

    with open("cs336_basics/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("cs336_basics/merges.pkl", "wb") as f:
        pickle.dump(merges, f)

from collections import defaultdict

corpus = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"

def vocab_init():
    vocab = {}
    for i in range(256):
        vocab[bytes([i])] = i
    vocab[b"<|endoftext|>"] = 256
    return vocab

def word_counter(words):
    word_freqs = defaultdict(int)
    for word in words:
        word_freqs[word] += 1
    return word_freqs

def word_to_tuple_of_bytes(word_freqs):
    bytes_freqs = {}
    for word, occurence in word_freqs.items():
        # Encode the word to bytes
        byte_data = word.encode("utf-8")
        # Use a comprehension to split them into single bytes objects
        single_bytes = tuple(bytes([b]) for b in byte_data)
        bytes_freqs[single_bytes] = occurence
    return bytes_freqs

def successive_pair_freq(bytes_freqs):
    pair_freqs = defaultdict(int)
    # Loop over each tuple
    for bytes_tuple, occurence in bytes_freqs.items():
        # Loop over each successive pair
        for i in range(len(bytes_tuple) - 1):
            # Add the occurence of that word to the pair's frequency
            pair_freqs[(bytes_tuple[i], bytes_tuple[i+1])] += occurence
    return pair_freqs

def merge_pair(bytes_freqs, pair):
    new_byte_freqs = {}
    # Walk through each tuple of bytes, merge bytes objects if they're pair
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


if __name__ == "__main__":
    # Init the vocab
    vocab = vocab_init()

    # Pre-tokenize the corpus
    pre_tokenized_words = corpus.split(" ")

    # Count each word's occurence
    word_freqs = word_counter(pre_tokenized_words)

    # Convert words to tuple of bytes 
    bytes_freqs = word_to_tuple_of_bytes(word_freqs)

    print("Original Byte freqs:", bytes_freqs)
    num_merges = 6
    for i in range(num_merges):
        print(f"Iteration {i+1}")

        # Count the frequency of each successive pair
        pair_freqs = successive_pair_freq(bytes_freqs)
        
        # Choose the most frequent pair. If tied,
        # take the alphabetically greater pair
        most_freq_pair = max(pair_freqs, key=lambda pair: (pair_freqs[pair], pair))
        print("Most frequent pair:", most_freq_pair)

        # Merge the most frequent pair and return the new bytes_freq dict
        bytes_freqs = merge_pair(bytes_freqs, most_freq_pair)
        print("New Byte freqs:", bytes_freqs)

        # Add the merged pair to our vocab
        vocab[most_freq_pair[0] + most_freq_pair[1]] = 257 + i
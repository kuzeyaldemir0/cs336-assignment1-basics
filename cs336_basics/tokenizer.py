import pickle
import regex as re

from collections.abc import Iterable, Iterator

class Tokenizer:

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a tokenizer from a given vocabulary, list of merges,
        and (optionally) a list of special tokens. This function should
        accept the following parameters:

        vocab: dict[int, bytes]

        merges: list[tuple[bytes, bytes]]

        special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.merges = merges
        self.reverse_vocab = {v:k for k, v in vocab.items()}

        if special_tokens is None:
            self.special_tokens = []
        else:
            special_tokens = sorted(special_tokens, key=lambda x: len(x), reverse=True)
            self.special_tokens = special_tokens
            for special_token in special_tokens:
                        encoded = special_token.encode("utf-8")
                        if encoded not in self.reverse_vocab:
                            self.reverse_vocab[encoded] = len(vocab)
                            vocab[len(vocab)] = encoded

        self.merge_priority_lookup = {}
        for idx, merge in enumerate(merges):
            self.merge_priority_lookup[merge] = idx

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        Class method that constructs and returns a Tokenizer from a
        serialized vocabulary and list of merges (in the same format
        that your BPE training code output) and (optionally) a list of
        special tokens. This method should accept the following
        additional parameters:

        vocab_filepath: str

        merges_filepath: str

        special_tokens: list[str] | None = None
        """
        with open(vocab_filepath, "rb") as file:
            vocab = pickle.load(file)

        with open(merges_filepath, "rb") as file:
            merges = pickle.load(file)

        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """

        if self.special_tokens:
            # Escape the '|' in special tokens
            escaped_special_tokens = []
            for sp_token in self.special_tokens:
                escaped_special_tokens.append(re.escape(sp_token))

            # Join all the special tokens with or ('|') operator for the regex pattern to comprehend that as OR operator
            special_token_pattern = "(" + "|".join(escaped_special_tokens) + ")"
            chunks = re.split(special_token_pattern, text)
        else:
            chunks = [text]

        result = []

        for chunk in chunks:
            if chunk in self.special_tokens:
                result.append([chunk.encode("utf-8")])
            else:   
                pretokenized = re.findall(self.PAT, chunk)
                for pretoken in pretokenized:
                    pretoken_bytes = pretoken.encode("utf-8")
                    pretoken_list_of_bytes = [bytes([b]) for b in pretoken_bytes]

                    i = 0
                    best_priority = len(self.vocab)
                    while i < len(pretoken_list_of_bytes) - 1:
                        pair = (pretoken_list_of_bytes[i], pretoken_list_of_bytes[i+1])
                        priority = self.merge_priority_lookup.get(pair)
                        if priority is not None:
                            if priority < best_priority:
                                best_priority = priority
                                merge_idx = i
                        i += 1
                        if i == len(pretoken_list_of_bytes) - 1 and best_priority < len(self.vocab):
                            i = 0
                            pretoken_list_of_bytes[merge_idx] = pretoken_list_of_bytes[merge_idx] + pretoken_list_of_bytes[merge_idx+1]
                            pretoken_list_of_bytes.pop(merge_idx+1)
                            best_priority = len(self.vocab)
                    result.append(pretoken_list_of_bytes)

        # Transform the list[list[bytes]] to list[int]
        final_result = []
        for bytes_list in result:
            for i in range(len(bytes_list)):
                byte = bytes_list[i]
                int_idx = self.reverse_vocab[byte]
                final_result.append(int_idx)

        return final_result


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return
        a generator that lazily yields token IDs. This is required for
        memory-efficient tokenization of large files that we cannot
        directly load into memory.
        """
        for string in iterable:
            for encoded_to_int in self.encode(string):
                yield encoded_to_int

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        # Given a list of ints, check the vocab and convert them to bytes and then decode using utf-8
        bytes_list = [
            self.vocab[id] for id in ids
        ]
        concat_bytes = b"".join(bytes_list)
        return concat_bytes.decode("utf-8", errors="replace")

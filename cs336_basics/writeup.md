## Problem (unicode1): Understanding Unicode

(a) '\x00'

(b) Its string representation is something like blank/null (not space), maybe created for not-existing strings so it doesn't show when printed compared to its representation given in question (a).

(c) It acts like nothing is there and probably something for null values that should be represented somehow.

##  Problem (unicode2): Unicode Encodings

(a) utf-8's main purpose is decreasing the beginning vocab size as small as possible from the original 153k vocab size of unicode encoding and utf-16 and utf-32 would have less of an advantage than utf-8 for that goal.

(b) to give an example the input "türkçe" would not work since some encodings correspond to multiple bytes but that function tries to decode each byte object in that list one by one which breaks non-ascii chars that contain multiple bytes such as the example given above.

(c) 0, 244 as byte one and two are not directly decodable when concat to a single bytes object.

## Problem (train_bpe_tinystories): BPE Training on TinyStories

(a) Don't know the memory usage but it worken on 24 GB of VRAM or unified memory macbook air m4. The training on the train set took 163 seconds, longest token is accomplishment, yeah if kind of makes sense considering the stories could be focusing on people achieving some stuff and they're all probably English.

(b) max method took 50 seconds of cumtime which is the max except the main functions and merge_pair for the case that took the longest from the helper functions we wrote took 2 seconds of cumulative time which is good considering we called it around 9k times and the whole profiling took 252 seconds.

## Problem (train_bpe_expts_owt):  BPE Training on OpenWebText

(a) Training took around 10k seconds, and peak memory was around 10gb. Longest token in the vocabulary is b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'

i guess it makes sense since i'm thinking OpenWebText might contain artifacts like these while they were scraping the web etc?

(b) The main difference is the OpenWebText vocab size is 32k while the TinyStories is 10k as specified on the problem sets. Other differences I observed are that the OpenWebText was not very clean like the TinyStroes and there were more interesting and unusual words/terms and more technical stuff perhaps in the final vocab especially close to the end.

## Problem (tokenizer_experiments):  Experiments with tokenizers

(a) TinyStories tokenizer → TinyStories sample: 4.0596 bytes/token. OpenWebText tokenizer → OpenWebText sample: 4.5966 bytes/token.

(b) OpenWebText tokenizer → TinyStories sample: 3.9945 bytes/token. TinyStories tokenizer → OpenWebText sample: 3.1908 bytes/token. I believe the content of the OpenWebText is more generic hence the compression ratio while using that instead of the TinyStories tokenizer on the Tiny Stories validation set didn't drop as much and also we couldn't neglect that the vocab size is 32k. Compression ratio while encoding the OpenWebText sample with the Tiny Stories tokenizer dropped a lot more and i'd explain it as both the smaller vocab and more "story-like" language which wouldn't have been able to encode the "web-like" text effectively as its own trained tokenizer perhaps.

(c) The throughput of our tokenizer in bytes/seconds on a 100 randomly sampled documents from the OpenWebText validation set was 1710680 bytes/second and at this rate i'd estimate the Pile Dataset (825 GB of text) would take around 134 hours.

(d) Our largest vocabulary has 32,000 tokens, so with 2^16 we have 0 to 65,535 possible values which would mean each token ID we have fits.

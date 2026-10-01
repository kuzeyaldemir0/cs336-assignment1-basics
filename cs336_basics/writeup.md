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

## Problem (transformer_accounting): Transformer LM resource accounting

(a) Trainable parameters for GPT-2 XL-sized model using our architecture:
1. token_embeddings: (vocab_size * d_model) $\to$ 50,257 * 1,600 = 80,411,200
    - transformer_block: ln1 + attn + ln2 + ffn $\to$ 1,600 + 10,240,000 + 1,600 + 20,582,400 = 30,825,600
2. layers: n_layers * transformer_block = 48 * 30,825,600 = 1,479,628,800
        - ln1: (d_model, ) $\to$ 1,600
        - attn: 4 * (d_model * d_model) $\to$ 4 * (1,600 * 1,600) = 10,240,000
        - ln2: (d_model, ) $\to$ 1,600
        - ffn: 3 * (d_model * d_ff) $\to$ 3 * (1,600 * 4,288) = 20,582,400
3. ln_final: (d_model, ) $\to$ 1,600
4. lm_head: (d_model * vocab_size) $\to$ 50,257 * 1,600 = 80,411,200

Total: 1,640,452,800 parameters, with single-precision (fp32) taking 4 bytes per parameter $\to$ 6,561,811,200 bytes just to store the parameters and therefore load the model, which would be close to 6,561.81 MB so near 6.56 GB.

(b) Matrix multiplications and total FLOPs of those matmuls for a forward pass of our GPT-2-XL-shaped model with the architecture we have. Assuming the input sequence has `context_length` tokens. T below is `context_length` and D is `d_model` and N is `n_layers` and V is `vocab_size`.
1. token_embeddings: no matmul computed.
2. layers: n_layers * transformer_block $\to$ 48 * 69,848,268,800 = 3,352,716,902,400 FLOPs
    - transformer_block: attn + ffn $\to$ 27,695,513,600 + 42,152,755,200 = 69,848,268,800 FLOPs
        - ln1: no matmul computed.
        - attn: 2 * T * D * (4*D + 2*T + 4) $\to$ 2 * 1,024 * 1,600 (4 * 1,600 + 2 * 1,024 + 4) = 27,695,513,600 FLOPs
        - ln2: no matmul computed.
        - ffn: 6 * T * D * d_ff $\to$ 6 * 1,024 * 1,600 * 4288 = 42,152,755,200 FLOPs
3. ln_final: no matmul computed.
4. lm_head: 2 * T * D * V $\to$ 2 * 1,024 * 1,600 * 50,257 = 164,682,137,600 FLOPs

Total: 3,517,399,040,000 FLOPs for a single forward pass with the specifications above. 

(c) The layers part of our model requires the most FLOPs and inside each of those layers, the attention and FFN computations requires the most FLOPs while attention requiring around 1.5x more FLOPs than FFN.

(d) Analysis for different sizes of the GPT-2 models, I only included the FLOPs for the modules that has matmul operations to keep things more compact:

1. GPT-2 small total FLOPs (12 layers, 768 d_model, 12 heads): layers + lm_head = 212,676,378,624 + 79,047,426,048 = **291,723,804,672** FLOPs
    1. layers: n_layers * transformer_block $\to$ 12 * 17,723,031,552 = 212,676,378,624
        - transformer_block: attn + ffn = 8,059,355,136 + 9,663,676,416 = 17,723,031,552
            - attn: 2 * T * D (4*D + 2*T + 4) $\to$ 2 * 1,024 * 768 (4 * 768 + 2 * 1,024 + 4) = 8,059,355,136
            - ffn: 6 * T * D * d_ff $\to$ 6 * 1,024 * 768 * 2,048 = 9,663,676,416
    2. lm_head: 2 * T * D * V $\to$ 2 * 1,024 * 768 * 50,257 = 79,047,426,048

Comments: Roughly 27% percent of the FLOPs belong to lm_head. The remaining 73% is from the layers module which consists of 12 transformer_block modules one after another, the attention part of the transformer_block consists 45% of the FLOPs in a single transformer_block and FFN has the remaining 55% roughly. So the attention's part of the whole transformer_lm is around 33% and FFN has 40% considering the FLOPs in the whole forward pass of the transformer_lm.

2. GPT-2 medium total FLOPs (24 layers, 1024 d_model, 16 heads): layers + lm_head = 724,977,057,792 + 105,396,568,064 = 830,373,625,856 FLOPs
    1. layers: n_layers * transformer_block $\to$ 24 * 30,207,377,408 = 724,977,057,792
        - transformer_block: attn + ffn $\to$ 12,893,290,496 + 17,314,086,912 = 30,207,377,408
            - attn: 2 * T * D (4*D + 2*T + 4) $\to$ 2 * 1,024 * 1,024 (4 * 1,024 + 2 * 1,024 + 4) = 12,893,290,496
            - ffn: 6 * T * D * d_ff $\to$ 6 * 1,024 * 1,024 * 2,752 = 17,314,086,912
    2. lm_head: 2 * T * D * V $\to$ 2 * 1,024 * 1,024 * 50,257 = 105,396,568,064 
 
Percentages: 
- Attention: $\approx$ 37.3%
- FFN: $\approx$ 50.0%
- LM Head: $\approx$ 12.7%

Comments: The percentage of the lm_head dropped to less than half the value it had in the small config while attn increased around 4 percentage points and FFN has also increased its total percentage.

3. GPT-2 large total FLOPs (36 layers, 1280 d_model, 20 heads): layers + lm_head = 1,637,162,680,320 + 131,745,710,080 = 1,768,908,390,400 FLOPs
    1. layers: n_layers * transformer_block $\to$ 36 * 45,476,741,120 = 1,637,162,680,320
        - transformer_block: attn + ffn $\to$ 18,800,967,680 + 26,675,773,440 = 45,476,741,120
            - attn: 2 * T * D (4*D + 2*T + 4) $\to$ 2 * 1,024 * 1,280 (4 * 1,280 + 2 * 1,024 + 4) = 18,800,967,680
            - ffn: 6 * T * D * d_ff $\to$ 6 * 1,024 * 1,280 * 3,392 = 26,675,773,440
    2. lm_head: 2 * T * D * V $\to$ 2 * 1,024 * 1,280 * 50,257 = 131,745,710,080

Percentages:
- Attention: $\approx$ 38.3%
- FFN: $\approx$ 54.3%
- LM Head: $\approx$ 7.4%

Comments: Attention and FFN increased slightly while LM Head kept dropping it's percentage to below 10%. So increasing the model size makes the lm_head's proportion to the total FLOPs less while increasing the proportion of Attention and FFN parts.

(e) Below are the comparison of the total FLOPs in a forward pass of the GPT-2-XL sized model with context length being 1,024 vs 16,384.

| Module | context_length = 1,024 | context_length = 16,384 |
| --- | ---: | ---: |
| Attention | $\approx 1.329e+12$ FLOPs ($\approx 37.79\%$) | $\approx 9.858e+13$ FLOPs ($\approx 73.79\%$) |
| FFN | $\approx 2.023e+12$ FLOPs ($\approx 57.52\%$) | $\approx 3.237e+13$ FLOPs ($\approx 24.23\%$) |
| LM Head | $\approx 1.647e+11$ FLOPs ($\approx 4.68\%$) | $\approx 2.635e+12$ FLOPs ($\approx 1.97\%$) |
| Total | $\approx 3.517e+12$ FLOPs ($\approx 100\%$) | $\approx 1.336e+14$ FLOPs ($\approx 100\%$) |

Comments: Attention reached almost the double percentage it had in the 1,024 context length when context length has increased to 16,384. FFN and LM head shares of the total FLOPs both dropped more than half.

## Problem (learning_rate_tuning):  Tuning the learning rate

1. 10 training steps, learning rate = 1e1, initial loss: 25.78, final loss = 3.46
2. 10 training steps, learning rate = 1e2, initial loss: 29.49, final loss = 3.51e-23
3. 10 training steps, learning rate = 1e3, initial loss: 23.59, final loss = 2.18e+18

As visible in the loss comparison above, the learning rate 1e1 decreases the loss successfully but not too fast. Learning rate 1e2 decreases the loss to almost 0 in the 10 steps and looks like the best learning rate from these 3 for this amount of steps and data. Learning rate 1e3 is too big and results in divergence which is basically loss increasing as seen in the final loss of that experiment.

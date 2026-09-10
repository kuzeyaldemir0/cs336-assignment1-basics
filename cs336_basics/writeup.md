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
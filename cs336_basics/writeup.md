## Problem (unicode1): Understanding Unicode

### (a) What Unicode character does chr(0) return?

Answer: '\x00'

### (b) How does this character’s string representation (__repr__()) differ from its printed representation?

Answer: Its string representation is something like blank/null (not space), maybe created for not-existing strings so it doesn't show when printed compared to its representation given in question (a).

### (c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:

Answer: It acts like nothing is there and probably something for null values that should be represented somehow.

## Problem (unicode2): Unicode Encodings

### (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

Answer: utf-8's main purpose is decreasing the beginning vocab size as small as possible from the original 153k vocab size of unicode encoding and utf-16 and utf-32 would have less of an advantage than utf-8 for that goal.

### (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'

Answer: to give an example the input "türkçe" would not work since some encodings correspond to multiple bytes but that function tries to decode each byte object in that list one by one which breaks non-ascii chars that contain multiple bytes such as the example given above.

### (c) Give a two-byte sequence that does not decode to any Unicode character(s).

Answer: 0, 244 as byte one and two are not directly decodable when concat to a single bytes object.
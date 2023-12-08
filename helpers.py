

def generate_pairs(word_list):
    pairs = [
        (word_list[i], word_list[j])
        for i in range(len(word_list))
        for j in range(i + 1, len(word_list))
    ]
    return pairs


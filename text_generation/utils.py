import random
import string


def generate_fake_text_sample(
        sample_len: int = 1000,
        word_min_len: int = 3,
        word_max_len: int = 10
) -> (str, int):

    current_len = 0
    result_text = ''

    while current_len <= sample_len:
        generated_word_len = random.randint(word_min_len, word_max_len)
        generated_word = ''.join([random.choice(string.ascii_letters) for i in range(generated_word_len)]) + ' '
        current_len += 1

        result_text += generated_word

    return result_text, current_len

t, l = generate_fake_text_sample()
print(t)
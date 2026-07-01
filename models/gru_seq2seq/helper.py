import tensorflow as tf
import regex as re

# Preprocess sentences - do not remove special characters from the sentences, as they are important for SQL queries
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    # sentence = re.sub(r'[" "]+', " ", sentence)
    # sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    sentence = sentence.strip()
    sentence = "<start> " + sentence + " <end>"
    return sentence


# allow for special characters by removing "strip" from the preprocess_sentence function
def create_dataset(path, num_examples=None):
    with open(path, encoding='utf-8') as f:
        lines = f.read().split('\n')
    sentence_pairs = [[preprocess_sentence(sentence) for sentence in line.split('\t')[:2]] for line in lines]
    #this returns a zip object of the first num_examples sentence pairs, which can be unpacked into source_lang and target_lang
    return zip(*sentence_pairs[:num_examples])

# Tokenize the data
def tokenize(lang):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(lang)
    tensor = tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, tokenizer


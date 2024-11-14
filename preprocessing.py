import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from pymorphy3 import MorphAnalyzer
from pyaspeller import YandexSpeller

nltk.download('punkt')
nltk.download('stopwords')

def to_lower(sentence):
    processed = [tkn.lower() for tkn in sentence]
    return processed


def tokenize(sentence):
    tokens = nltk.word_tokenize(sentence, language='russian')
    return tokens


def en_ru_mapping(sentence):
    mapper = {
        'o': 'о', 'i': 'и', 'k': 'к', 'x': 'х',
        'r': 'р', 'a': 'а', 'e': 'е', 'c': 'с',
        'y': 'у', 'p': 'р', 't': 'т', 'm': 'м',
        'l': 'л', 'n': 'н', 's': 'с', 'v': 'в',
        'b': 'б', 'd': 'д', 'u': 'у', 'g': 'г',
        'h': 'х', 'w': 'в', 'q': 'г'
    }
    processed = [''.join([mapper.get(chr, chr) for chr in tkn]) for tkn in sentence]
    return processed


def replace_numbers(sentence):
    processed = []
    for tkn in sentence:
        # 1. Замена '3' на 'з' после гласной или в начале слова
        tkn = re.sub(r'\b3', 'з', tkn)  # В начале слова
        tkn = re.sub(r'([аеёиоуыэюяАЕЁИОУЫЭЮЯ])3', r'\1з', tkn)  # После гласной

        # 2. Замена '3' на 'е' после согласной
        tkn = re.sub(r'([бвгджзклмнпрстфхцчшщБВГДЖЗКЛМНПРСТФХЦЧШЩ])3', r'\1е', tkn)

        # 3. Замена '0' на 'о' внутри слова или на его границах
        tkn = re.sub(r'\b0(?=\w)|(?<=\w)0\b|(?<=\w)0(?=\w)', 'о', tkn)

        # 4. Замена '1' на 'и' внутри слова
        tkn = re.sub(r'(?<=\w)1(?=\w)', 'и', tkn)

        # 5. Замена '7' на 'т' внутри слова или на его границах
        tkn = re.sub(r'\b7(?=\w)|(?<=\w)7\b|(?<=\w)7(?=\w)', 'т', tkn)

        processed.append(tkn)

    return processed


def only_ru_chars(sentence):
    processed = []
    for tkn in sentence:
        cleaned = re.sub(r'[^А-Яа-яЁё\s]', '', tkn)
        if cleaned != '':
            processed.append(cleaned)
    return processed


def remove_repeating_letters(sentence):
    pattern = r'([А-Яа-яЁё])\1+'
    processed = []
    for tkn in sentence:
        processed.append(re.sub(pattern, r'\1', tkn))
    return processed


def remove_stop_words(sentence):
    stopwords = nltk.corpus.stopwords.words("russian")
    processed = [tkn for tkn in sentence if tkn not in stopwords]
    return processed


def stemming(sentence):
     stemmer = SnowballStemmer("russian")
     processed = [stemmer.stem(tkn) for tkn in sentence]
     return processed


def remove_freqs(text):
    tokens = [word for sentence in text for word in sentence]
    fdist = nltk.FreqDist(tokens)
    stop_wrds = [
        'скаи', 'скай', 'скать', 'сказть', 'сказт', 'вобще', 'боле', 'скаь', 'скайть', 'ктото',
        'както', 'оно', 'этим', 'какими', 'сами', 'другим', 'ве', 'eщ', 'сам', 'из', 'наш' , 'бо', 'ска', 'сказыв',
        'хто', 'цк', 'дон', 'качть', 'мур', 'такая', 'будь', 'что', 'чан', 'кас', 'буд', 'будт', 'есл', 'ещ', 'иза', 'уге',
        'тыс', 'ида', 'ет', 'ед', 'сво', 'хий', 'ом', 'ещё', 'вроде', 'который', 'тип'
    ]
    processed = []
    for sentence in text:
        s = ' '.join([tkn for tkn in sentence if ((fdist[tkn] < fdist.N() * 0.005) and (tkn not in stop_wrds))])
        processed.append(s)
    return processed


def lemmatize(sentence):
    morph = MorphAnalyzer()
    processed = [morph.parse(tkn)[0].normal_form for tkn in sentence]
    return processed


def spell_correction(sentence):
    speller = YandexSpeller(find_repeat_words=True)
    processed = speller.spelled(' '.join(sentence)) 
    return processed
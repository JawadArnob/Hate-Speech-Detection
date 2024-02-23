import warnings
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import contractions
import nltk
import re
import difflib
import pandas


class TextProcessor:
    WORD_WITH_DIGIT_PATTERN = re.compile(r'/\b[a-zA-Z]\w*\d\w*[a-zA-Z]\b/gm')
    HASHTAG_PATTERN = re.compile('(\#[a-zA-Z0-9]+)(?!;)')
    HASHTAG_PARSE_PATTERN = re.compile('[A-Z][^A-Z]*')
    URL_PATTERN = re.compile(r'((http|https|ftp|ftps)\:\/\/)?[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(\/\S*)?')
    TOKENIZER = nltk.RegexpTokenizer(r'(?u)\b\w\w+\b')
    MENTION_TAG_PATTERN = re.compile(
        r"(?<![A-Za-z0-9_!@#\$%&*])@"
        r"(([A-Za-z0-9_]){15}(?!@)|([A-Za-z0-9_]){1,14}(?![A-Za-z0-9_]*@))"
    )
    LEMMATIZER = nltk.WordNetLemmatizer()
    ABBREVIATIONS = {
        'u': 'you',
        'btch': 'bitch',
        'bf': 'boy-friend',
        'un': 'united-nations',
        'yo': 'you',
        'wtf': 'what the fuck',
        'gvt': 'government',
    }
    DIGIT_PLACEHOLDERS: Dict[str, str] = {
        '1': 'i',
        '4': 'a',
        '5': 's',
        '6': 'g',
        '0': 'o',
    }
    IMPORTANT_EMOJIS = {
        "🖕": "dick",
        "😉": "winkingface",
        "😏": "winkingface",
        "😷": "maskface",
        "🌵": "dick",
        "❌": "not",
        "🤧": "sneezing",
        "🤬": "angry",
        "💦": "ejaculates",
        "😡": "angry",
        "😠": "angry",
        "😤": "angry",
        "😈": "evil",
        "😂": "laugh",
        "💪": "fist",
        "👊": "fist",
        "✊": "fist",
        "😝": "tongueface",
        "🤘🏼": "horns",
        "💀": "evil",
        "🤢": "nasty",
        "😍": "smiling",
        "🍆": "dick",
        "👅": "suck",
        "💃🏻": "danc",
        "🤤": "thirsty",
        "🚮": "trash",
        "🙂": "smile",
        "🙄": "think",
        "😒": "sad",
        "😛": "suck",
        "🔥": "hot",
        "❗️": "exclamation",
        "💩": "shit",
        "👏": "slap",
    }
    IMPORTANT_WORDS: List[str] = [
        'bitch',
        'woman',
        'illegal',
        'whore',
        'immigration',
        'fuck',
        'refugee',
        'country',
        'cunt',
        'migrant',
        'people',
        'stop',
        'fucking',
        'trump',
        'immigrant',
        'back',
        'rape',
        'girl',
        'illegals',
        'alien',
        'border',
        'shit',
        'home',
        'stupid',
        'hysterical',
        'dick',
        'skank',
        'pussy',
        'come',
        'right',
        'europe',
        'never',
        'must',
        'child',
        'white',
        'even',
        'america',
        'wall',
        'deport',
        'american',
        'life',
        'law',
        'look',
        'americans',
        'muslim',
        'world',
        'keep',
        'slut',
        'every',
        'another',
        'dumb',
        'nigga',
        'hate',
        'please',
        'crime',
        'illegalaliens',
        'stay',
        'going',
        'enough',
        'money',
        'care',
        'vote',
        'stoptheinvasion',
        'open',
        'germany',
        'better',
        'enddaca',
        'criminal',
        'suck',
        'illegally',
    ]
    STOPWORDS = set(nltk.corpus.stopwords.words('english'))
    KEEP_WORDS = ['do','he','his','him','herself','themselves','who','those','but','against','here','yourself','our','about','as','soon','because','out','get','not','what','will', 'you', 'your', 'should', 'she', 'they', 'their', 'them', 'her', 'my', 'have', 'need', 'has', 'will']
    STOPWORDS.difference_update(KEEP_WORDS)
    PUNCTUATION = r"""!"#$%&'()*+,-./:;<=…>?@[\]^_`{|}...~&amp;"""
    EMOJI_DATA = None
    EMOJI_PATTERN = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    def __init__(
            self,
            digit_replacer: Dict[str, str] = None,
            abbreviations: Dict[str, str] = None,
            emoji_data: Dict[str, str] = None,
            important_words: List[str] = None,
    ):
        """
        To remove hashtag user <TextProcessor>.remove_hashtags = True
        Other attributes with default values are:
        self.remove_stopwords = True
        self.replace_hashtags = True
        self.remove_hashtags = False
        self.replace_urls = True
        self.replace_digits = True
        self.replace_emojis = True
        self.lemmatize = False
        self.replace_abbreviations = True
        self.replace_contractions = True
        self.remove_mentions = False

        :param digit_replacer: Dict Sample: {'1': 'i'}
        :param abbreviations: Dict Sample: {'u': 'you'}
        :param emoji_data: Please use the EMOJI_DATA of emoji_data provided in the package
        :param important_words: List of words that are important to the model
        """
        if digit_replacer:
            self.DIGIT_PLACEHOLDERS = digit_replacer
        if abbreviations:
            self.ABBREVIATIONS = abbreviations
        if emoji_data:
            self.EMOJI_DATA = emoji_data
        if important_words:
            self.IMPORTANT_WORDS = important_words
            self.similarities_check = True
        else:
            self.similarities_check = True  # for testing
            # self.similarities_check = False
        self.remove_stopwords = True
        self.replace_hashtags = True
        self.remove_hashtags = False
        self.replace_urls = True
        self.replace_digits = True
        self.replace_emojis = True
        self.lemmatize = False
        self.replace_abbreviations = True
        self.replace_contractions = True
        self.remove_mentions = False

    def process_dataframe(self, dataframe: pandas.DataFrame, column_key: str) -> pandas.DataFrame:
        for index in dataframe.index:
            setattr(dataframe.iloc[index], column_key, self.process_line(getattr(dataframe.iloc[index], column_key)))
        return dataframe

    def process_plain_text(self, plain_text: str, dataframe_output: bool = False) -> Optional[Union[str, pandas.DataFrame]]:
        processed_text = self.process_line(plain_text)
        if dataframe_output:
            return pandas.DataFrame.from_records([{'text': processed_text}])
        return processed_text

    def process_line(self, line_text: str) -> str:
        
        line_text = re.sub(r'@\S+', ' user ', line_text)
        line_text = re.sub(r'(user\s+){2,}', '  user ', line_text)
        line_text = re.sub(r'(\?){2,}', ' question ', line_text)
        line_text = re.sub(r'(!){2,}', ' exclamation ', line_text)
        line_text = re.sub(r'\?', '  question ', line_text)
        line_text = re.sub(r'!', '  exclamation ', line_text)
        line_text =  re.sub(r'\d+', '', line_text) #<------------ I added number removal

        # Replace Contractions
        if self.replace_contractions:
            line_text = contractions.fix(line_text)

        # Replace URLs
        if self.replace_urls:
            line_text = line_text.replace('http', ' http')
            line_text = self.URL_PATTERN.sub('', line_text)

        # Replace Digits in Word (i.e )
        if self.replace_digits:
            words_with_digits = self.WORD_WITH_DIGIT_PATTERN.findall(line_text)
            for word in words_with_digits:
                line_text = line_text.replace(word, self.digit_replacer(word))

        # Parse Hashtags
        if self.replace_hashtags:
            hashtags = self.HASHTAG_PATTERN.findall(line_text)
            for hashtag in hashtags:
                line_text = line_text.replace(hashtag, self.parse_hashtag(hashtag))

        # Replace Emojis
        if self.replace_emojis:
            line_text = self.EMOJI_PATTERN.sub(self.replace_emoji, line_text)

        # E-[word] Replacement
        line_text = re.sub(r"[eE]-(\w+)", r"e\1", line_text)

        # Remove Mentions (@user)
        if self.remove_mentions:
            line_text = self.MENTION_TAG_PATTERN.sub('', line_text)

        # Tokenize and Process Tokens
        line_text = line_text.lower()
        processed_tokens: List[str] = []
        for token in self.TOKENIZER.tokenize(line_text):
            # Remove Stopwords
            if self.remove_stopwords and token in self.STOPWORDS:
                #processed_tokens.append(token) #<----------------- fix remove_stopwords was not remved so i comment this
                continue

            # Replace Abbreviations
            if self.replace_abbreviations and token in self.ABBREVIATIONS:
                processed_tokens.extend(self.get_abbreviations(token))
                continue

            # Remove Punctuations/Special Characters
            if token in self.PUNCTUATION:
                continue

            # Spellcheck for Important Words
            if self.similarities_check:
                similarities = self.get_similarities(token)
                if similarities:
                    #processed_tokens.extend(similarities) 
                    processed_tokens.append(self.LEMMATIZER.lemmatize(similarities[0], pos='v')) #<--- fix

                    continue

            # Lemmatize Words
            if self.lemmatize:
                #processed_tokens.append(self.LEMMATIZER.lemmatize(token))
                processed_tokens.append(self.LEMMATIZER.lemmatize(token, pos='v')) #<---- fix

            else:
                processed_tokens.append(token)

        return ' '.join(processed_tokens)

    def get_similarities(self, token: str) -> List[str]:
        temp_res = []
        clean_token = re.sub(r'(.)\1+', r'\1', token)
        for important_word in self.IMPORTANT_WORDS:
            if important_word in token or important_word in clean_token:
                temp_res.append(important_word)
        if temp_res:
            return temp_res

        ratios = [difflib.SequenceMatcher(None, clean_token, word).ratio() for word in self.IMPORTANT_WORDS]
        if max(ratios) >= 0.8:
            closest_match_index = ratios.index(max(ratios))
            return [self.IMPORTANT_WORDS[closest_match_index]]
        else:
            return []

    def get_abbreviations(self, token: str) -> List[str]:
        return list(filter(lambda t: t not in self.STOPWORDS, self.ABBREVIATIONS[token].split(' ')))

    def digit_replacer(self, word: str) -> str:
        for char, replacement in self.DIGIT_PLACEHOLDERS.items():
            word = word.replace(char, replacement)
        return word.lower()

    def parse_hashtag(self, text: str) -> str:
        if self.remove_hashtags:
            return ''
        text = text.replace('#', '')
        regex_result = self.HASHTAG_PARSE_PATTERN.findall(text)
        if not regex_result:
            return text
        if len(regex_result) > len(text) * 0.5:
            return text
        return ' '.join(regex_result)

    def replace_emoji(self, match) -> str:
        res = ''
        for emoji in match.group():
            if emoji in self.IMPORTANT_EMOJIS:
                res += self.IMPORTANT_EMOJIS[emoji]
            else:
                res += self.EMOJI_DATA.get(emoji, {}).get('en', ':').replace(':', '').lower()
        return res

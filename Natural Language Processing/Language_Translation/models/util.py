import unicodedata
import re


def lower_and_split_punct(text):
    '''
    text cleaning 
    '''
    # Split accented characters (normalize to NFKD form)
    text = unicodedata.normalize('NFKD', text)

    # Convert text to lowercase
    text = text.lower()

    # Keep spaces, a to z, and select punctuation, and remove other characters like '\u202f' and numbers
    text = re.sub(r'[^ a-z.?!,¿]', '', text)

    # Add spaces around punctuation
    text = re.sub(r'([.?!,¿])', r' \1 ', text)

    # Strip extra whitespace
    text = text.strip()

    # Add [START] and [END] tokens
    text = '[START] ' + text + ' [END]'

    return text

def Max_length(data):
    '''
    finding max length for the sequence
    '''
    max_length_ = max([len(x.split(' ')) for x in data])
    return max_length_



def target_sequence(input_seq,Fword2index,Findex2word):
    '''
    target sequence post translation
    '''
    newString=''
    for i in input_seq:
      if((i!=0 and i!=Fword2index['start']) and i!=Fword2index['end']):
        newString=newString+Findex2word[i]+' '
    return newString

def source_sequence(input_seq,Eindex2word):
    '''
    source sequence 
    '''
    newString=''
    for i in input_seq:
      if(i!=0):
        newString=newString+Eindex2word[i]+' '
    return newString
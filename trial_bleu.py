from nltk.translate.bleu_score import sentence_bleu

reference = [
    'this is a dog',
    'it is dog',
    'dog it is',
    'a dog, it is'
]

print(reference)
candidate = 'it is dog'
print('BLEU score -> {}'.format(sentence_bleu(reference, candidate)))

candidate = 'it is a dog'.split()
print('BLEU score -> {}'.format(sentence_bleu(reference, candidate)))
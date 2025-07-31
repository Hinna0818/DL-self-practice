import sys
sys.path.append('..')
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.utils import preprocess, create_contexts_target, convert_one_hot

window_size = 1
hiden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word2id, id2word = preprocess(text)

vocab_size = len(word2id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hiden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot(ylim=(0, 1.5))

## Word Vectors
word_vecs = model.word_vecs
for word_id, word in id2word.items():
    print(f'word: {word}, vector: {word_vecs[word_id]}')
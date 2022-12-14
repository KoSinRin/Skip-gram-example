import numpy as np
from keras.layers import Embedding, Dot, Input
from keras.models import Model

# Размерность векторных представлений слов
embedding_dim = 100

# Размер окна контекста
window_size = 2

# Создаем модель
input_word = Input(shape=(1,))
context_words = Input(shape=(window_size*2,))

# Слой эмбеддингов
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1)(input_word)

# Слой умножения эмбеддинга центрального слова на эмбеддинги контекстных слов
dot = Dot(axes=2)([embedding, context_words])

# Объединяем слои в модель
skip_gram_model = Model(inputs=[input_word, context_words], outputs=dot)

# Компилируем модель
skip_gram_model.compile(loss='binary_crossentropy', optimizer='adam')

# Обучаем модель на данных
skip_gram_model.fit([input_words, context_words], labels, epochs=10, batch_size=32)















import pandas as pd
import json
from smart_open import open
from pprint import pprint

ds = []
with open('data/comment.json', 'r') as f:
    # print(len(f.readlines()))
    for line in f.readlines():
        d = json.loads(line)
        ds.append(d)

pprint(ds[:10])

a = pd.Series([
    d['content']
    for d in ds
])


pprint(a.apply(len).max())
pprint(a.apply(len).min())
pprint(a.apply(len).mean())
pprint(a.apply(len).median())

import matplotlib.pyplot as plt
plt.hist(a.apply(len), bins=150)
plt.show()

a = list(a)
train_data_path = "train_data.txt"
model_path = "example.model"
with open(train_data_path, 'w') as f:
    f.write('\n'.join(a))


import youtokentome as yttm
yttm.BPE.train(data=train_data_path, vocab_size=15000, model=model_path)

# Loading model
bpe = yttm.BPE(model=model_path)

test = [
    'One thing I want to ask:'
    ' why clarkson mistake! '
    'One thing everyone'
    ' can agree on, this game'
    ' is way better than we'
    ' every'
]
# Two types of tokenization
print(
    bpe.encode(
        test,
        output_type=yttm.OutputType.ID
    )
)
print(
    bpe.encode(
        test,
        output_type=yttm.OutputType.SUBWORD
    )
)




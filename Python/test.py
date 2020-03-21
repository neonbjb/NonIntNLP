import torch
import transformers

tok = transformers.XLNetTokenizer.from_pretrained("xlnet-base-cased")
v = torch.load('C:/Users/jbetk/Documents/data/ml/title_prediction/classification_outputs/val.pt')
for e in v:
    print(tok.decode(e['text'].tolist()))
    print(tok.decode(e['target'].tolist()))
    print(e['classifier'])
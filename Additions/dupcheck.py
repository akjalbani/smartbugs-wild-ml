import pickle, hashlib, json, numpy as np
from src import features
d=pickle.load(open('stage/split.pkl','rb')); kept=d['kept']
H=[hashlib.sha1(" ".join(features.tokenize(features.read_contract(a))).encode()).hexdigest() for a in kept]
tr=set(H[i] for i in d['tr']); te=[H[i] for i in d['te']]
dupmask=np.array([x in tr for x in te])
out={"unique":len(set(H)),"total":len(H),"test_in_train":int(dupmask.sum()),"n_test":len(te)}
np.save('stage/test_dup_mask.npy',dupmask); json.dump(out,open('stage/dup.json','w')); print(out)

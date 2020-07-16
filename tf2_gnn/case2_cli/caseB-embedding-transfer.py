import pandas as pd
pd.set_option('display.max_rows', 45)
df = pd.read_csv(r"data_caseB/pact-2014-runtimes.csv")
oracles = pd.read_csv(r"data_caseB/pact-2014-oracles.csv")
import numpy as np
class ThreadCoarseningModel(object):
    __name__ = None
    __basename__ = None

    def init(self, seed: int) -> None:
        pass

    def save(self, outpath: str) -> None:
        raise NotImplementedError

    def restore(self, inpath: str) -> None:
        raise NotImplementedError

    def train(self, sequences: np.array, y_1hot: np.array, verbose: bool=False) -> None:
        raise NotImplementedError

    def predict(self,sequences: np.array) -> np.array:
        raise NotImplementedError

cfs = [1, 2, 4, 8, 16, 32]  # thread coarsening factors

def get_onehot(df, platform):
    hot = np.zeros((len(df), len(cfs)), dtype=np.int32)
    for i, cf in enumerate(df[f"cf_{platform}"]):
        hot[i][cfs.index(cf)] = 1
    return hot

def get_features(df, oracles, platform):
    """
    Assemble cascading data.
    """
    X_cc, y_cc, = [], []
    for kernel in sorted(set(df["kernel"])):
        _df = df[df["kernel"] == kernel]

        oracle_cf = int(oracles[oracles["kernel"] == kernel][f"cf_{platform}"].values[0])

        feature_vectors = np.asarray([
            _df['PCA1'].values,
            _df['PCA2'].values,
            _df['PCA3'].values,
            _df['PCA4'].values,
            _df['PCA5'].values,
            _df['PCA6'].values,
            _df['PCA7'].values,
        ]).T
                
        X_cc.append(feature_vectors)
        y = []
        cfs__ = []
        for i, cf in enumerate(cfs[:len(feature_vectors)]):
            y_ = 1 if cf < oracle_cf else 0
            y.append(y_)
        y_cc.append(y)
    
        assert len(feature_vectors) == len(y)
        
    assert len(X_cc) == len(y_cc) == 17
    
    return np.asarray(X_cc), np.asarray(y_cc)


def encode_srcs(srcs):
    """ encode and pad source code for learning """
    from keras.preprocessing.sequence import pad_sequences
    
    seqs = [atomizer.atomize(src) for src in srcs]
    pad_val = atomizer.vocab_size
    encoded = np.array(pad_sequences(seqs, maxlen=1024, value=pad_val))
    return np.vstack([np.expand_dims(x, axis=0) for x in encoded])


def platform2str(platform):
    if platform == "Fermi":
        return "NVIDIA GTX 480"
    elif platform == "Kepler":
        return "NVIDIA Tesla K20c"
    elif platform == "Cypress":
        return "AMD Radeon HD 5900"
    elif platform == "Tahiti":
        return "AMD Tahiti 7970"
    else:
        raise LookupError

# 存放一些工具文件
import os
import random

import numpy as np
import json
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from keras.utils.np_utils import to_categorical
TRUE = [0, 1]
FALSE = [1, 0]
def get_file_name(data_dir):
    import glob
    file_name = []
    for childDir in os.listdir(data_dir):
        if (childDir != "caseb_json_new"):
            continue
        new_path = os.path.join(data_dir, childDir)
        lens = len(os.listdir(new_path))
        print("file number:", lens)
        name=os.listdir(new_path)
        number=[]
        for i in range(len(name)):
            number.append(int(name[i][:-5]))
        numbers = sorted(number, reverse = False)
        for i in numbers:
            file=str(i)+".json"
            file_name.append(os.path.join(new_path, file))
    return file_name


class graph(object):
    def __init__(self, node_num=0, label=None, name=None):
        
        self.node_num = node_num
       
        self.label = label
        self.name = name
        
        self.features = []
       
        self.child = []
        self.next_token = []
        self.compute_from = []
        self.guarded = []
        self.jump = []
        self.lexical_use = []
 
        self.preds = []
       
        if (node_num > 0):
            for i in range(node_num):
                self.features.append([])
                self.child.append([])
                self.next_token.append([])
                self.compute_from.append([])
                self.guarded.append([])
                self.jump.append([])
                self.lexical_use.append([])

 
    def add_node(self, feature=[]):
        self.node_num += 1
        self.features.append(feature)
        self.succs.append([])
        self.preds.append([])

    def add_child_edge(self, u, v):
        self.child[u].append(v) 
    def add_next_token_edge(self, u, v):
        self.next_token[u].append(v)
    def add_compute_from_edge(self, u, v):
        self.compute_from[u].append(v)
    def add_guarded_edge(self, u, v):
        self.guarded[u].append(v)
    def add_jump_edge(self, u, v):
        self.jump[u].append(v)
    def add_lexical_use_edge(self, u, v):
        self.lexical_use[u].append(v)
#         self.preds[v].append(u)

   
    def toString(self):
        ret = '{} {}\n'.format(self.node_num, self.label)
        for u in range(self.node_num):
            for fea in self.features[u]:
                ret += '{} '.format(fea)
            ret += str(len(self.succs[u]))
            for succ in self.succs[u]:
                ret += ' {}'.format(succ)
            ret += '\n'
        return ret


def read_graph(F_NAME, Use_Self_Fea):
    graphs = []
    classes = []
    tc = 0
    fc = 0
    for f_name in F_NAME:
        with open(f_name) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                label = TRUE
                classes.append(len(graphs))
                cur_graph = graph(int(g_info['child_num']), label, f_name)
                for u in range(int(g_info['child_num'])):
                    if Use_Self_Fea:
                        cur_graph.features[u] = np.array(g_info['featureDims'][u])
                        
                    else:
                        cur_graph.features[u] = np.array(g_info['features'][u])
                        
                    for v in g_info['child'][u]:
                        cur_graph.add_child_edge(u, v)
                    for v in g_info['next_token'][u]:
                        cur_graph.add_next_token_edge(u, v)
                    for v in g_info['compute_from'][u]:
                        cur_graph.add_compute_from_edge(u, v)
                    for v in g_info['guarded'][u]:
                        cur_graph.add_guarded_edge(u, v)
                    for v in g_info['jump'][u]:
                        cur_graph.add_jump_edge(u, v)
                    for v in g_info['lexical_use'][u]:
                        cur_graph.add_lexical_use_edge(u, v)
                graphs.append(cur_graph)

    return graphs, classes




seed = 204

import pickle
import sys
from labm8 import fs
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold

def evaluate(model):
    from progressbar import ProgressBar
    progressbar = [0, ProgressBar(maxval=68)]
    progressbar[1].start()
    data = []
    
    X_seq = None  # defer sequence encoding (it's expensive)
    
    for i, platform in enumerate(["Cypress", "Tahiti", "Fermi", "Kepler"]):
        platform_name = platform2str(platform)
                
        # 读取四个平台下标签的运行时
        oracle_runtimes = np.array([float(x) for x in oracles["runtime_" + platform]])
        # 读取四个平台下的标签（粗化因子）
        y = np.array([int(x) for x in oracles["cf_" + platform]], dtype=np.int32)
        # 对标签6种情况一热编码
        y_1hot = get_onehot(oracles, platform)
        X_cc, y_cc = get_features(df, oracles, platform)
        embed = np.load("data_caseB/caseb_128.npy")

        
        kf = KFold(n_splits=len(y), shuffle=False)
    
        for j, (train_index, test_index) in enumerate(kf.split(y)):
            kernel = sorted(set(df["kernel"]))[test_index[0]]
            
            model_name = model.__name__
            model_basename = model.__basename__
            
            model_path = f"result_caseB/modelb_caseB/{model_basename}-{platform}-{j}.model"
            predictions_path = f"result_caseB/predictionb_caseB/{model_basename}-{platform}-{j}.result"  

            if fs.exists(predictions_path):
                # load result from cache
                with open(predictions_path, 'rb') as infile:
                    p = pickle.load(infile)
            else:
                if fs.exists(model_path):
                    # load a trained model from cache
                    model.restore(model_path)
                else:
                    
                    # create a new model and train it
                    model.init(seed=seed)
                    model.train(sequences=embed[train_index],
                                verbose=True, # TODO
                                y_1hot=y_1hot[train_index])

                    # cache the model
                    fs.mkdir(fs.dirname(model_path))
                    model.save(model_path)
                # make prediction
                p = model.predict(sequences=np.array(embed[test_index[0]]))[0]
      
                p = min(p, 2 ** (len(X_cc[test_index[0]]) - 1))
                
                # cache the prediction
                fs.mkdir(fs.dirname(predictions_path))
                with open(predictions_path, 'wb') as outfile:
                    pickle.dump(p, outfile)
                    
            # oracle prediction
            o = y[test_index[0]]
            correct = p == o
            # get runtime without thread coarsening
            row = df[(df["kernel"] == kernel) & (df["cf"] == 1)]
            assert(len(row) == 1)  # sanity check
            nocf_runtime = float(row["runtime_" + platform])

            # get runtime of prediction
            row = df[(df["kernel"] == kernel) & (df["cf"] == p)]
            assert(len(row) == 1)  # sanity check
            p_runtime = float(row["runtime_" + platform])
            
            # get runtime of oracle coarsening factor
            o_runtime = oracle_runtimes[test_index[0]]
            # speedup and % oracle
            s_oracle = nocf_runtime / o_runtime
            p_speedup = nocf_runtime / p_runtime
            p_oracle = o_runtime / p_runtime

            # record result
            data.append({
                "Model": model_name,
                "Platform": platform_name,
                "Kernel": kernel,
                "Oracle-CF": o,
                "Predicted-CF": p,
                "Speedup": p_speedup,
                "Oracle": p_oracle
            })
            
            progressbar[0] += 1  # update progress bar
            progressbar[1].update(progressbar[0])

    return pd.DataFrame(data, columns=[
        "Model", "Platform", "Kernel", "Oracle-CF", "Predicted-CF", "Speedup", "Oracle"])


# In[8]:


from keras.layers.core import Layer
from keras.engine import InputSpec
from keras import backend as K
import tensorflow as tf 
try:
    from keras import initializations
except ImportError:
    from keras import initializers as initializations

class GNN(Layer):
    def __init__(self,NODE_FEATURE_DIM,N_embed,depth_embed,N_output,ITER_LEVEL,lr,Dtype=tf.float32, **kwargs):
        
        self.NODE_FEATURE_DIM = NODE_FEATURE_DIM
        self.Dtype = Dtype
        self.N_embed = N_embed
        self.depth_embed = depth_embed
        self.N_output=N_output
        self.ITER_LEVEL = ITER_LEVEL
        self.lr = lr
        
        super(GNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.Wnode = self.add_weight(shape=(self.NODE_FEATURE_DIM, self.N_embed),initializer = 'truncated_normal',trainable = True,name='{}_Wnode'.format(self.name))
        self.Wembed = []
        for i in range(self.depth_embed):
            self.Wembed.append(self.add_weight(shape=(self.N_embed, self.N_embed),initializer = 'truncated_normal',trainable = True,name='{}_Wembed_{}'.format(self.name,i)))
        self.W_output = self.add_weight(shape=(self.N_embed, self.N_output),initializer = 'truncated_normal',trainable = True,name='{}_W_output'.format(self.name))
        super(GNN, self).build(input_shape)
    

    def call(self, x, mask=None):
        msg1_mask = x[0]
        X1 = x[1]
        
        node_val = K.reshape(tf.matmul(K.reshape(X1, [-1, self.NODE_FEATURE_DIM]), self.Wnode),
                                   [K.shape(X1)[0], -1, self.N_embed])
        cur_msg = tf.nn.relu(node_val)   # [batch, node_num, embed_dim]
        for t in range(self.ITER_LEVEL):
            # Message convey
            Li_t = tf.matmul(msg1_mask, cur_msg)  # [batch, node_num, embed_dim]
            # Complex Function  
            cur_info = K.reshape(Li_t, [-1, self.N_embed])
            for Wi in self.Wembed:
                if Wi == self.Wembed[-1]:
                    cur_info = tf.matmul(cur_info, Wi)
                else:
                    cur_info = tf.nn.relu(tf.matmul(cur_info, Wi))
            neigh_val_t = K.reshape(cur_info, K.shape(Li_t))
            # Adding
            tot_val_t = node_val + neigh_val_t
            # NonLinearity
            tot_msg_t = tf.nn.tanh(tot_val_t)
            cur_msg = tot_msg_t   # [batch, node_num, embed_dim]
        g_embed = tf.reduce_sum(cur_msg, 1)   # [batch, embed_dim]
        out =tf.matmul(g_embed, self.W_output)

        
        return out
    def compute_output_shape(self,input_shape): 
        features_shape = input_shape[1]
        output_shape = (features_shape[0], self.N_output)
        return output_shape
    
    def get_output_shape_for(self, input_shape):
        features_shape = input_shape[1]
        output_shape = (features_shape[0], self.N_output)
        return output_shape


# In[9]:


class RGIN(ThreadCoarseningModel):
    __name__ = "RGIN"
    __basename__ = "rgin"

    def init(self, seed: int=None):
        from keras.layers import Input, Dropout, Embedding, merge, LSTM, Dense
        from keras.layers.normalization import BatchNormalization
        from keras.models import Model, Sequential, load_model
    
        np.random.seed(seed)
        embedding = Input(shape=(256,), dtype="float32", name="embedding")
        x = Dense(32, activation="relu")(embedding)
        outputs = Dense(6, activation="sigmoid")(x)
        
        self.model = Model(inputs=embedding, outputs=outputs)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    def save(self, outpath: str):
        self.model.save(outpath)

    def restore(self, inpath: str):
        self.model = load_model(inpath)
    
    
    def train(self, sequences: np.array, y_1hot: np.array, verbose: bool=False) -> None:
        self.model.fit(sequences, y_1hot, epochs=50, batch_size=64, verbose=verbose, shuffle=True)
        
    def predict(self,sequences: np.array) -> np.array:
        # directly predict optimal thread coarsening factor from source sequences:
        sequences = np.array([sequences])
        # print(sequences.shape)
        print("sequences's length=",len(sequences))
        p = np.array(self.model.predict(sequences, batch_size=64, verbose=0))
        indices = [np.argmax(x) for x in p]
        return [cfs[x] for x in indices]
    


# In[10]:


rgin_model = RGIN()
rgin_model.init(seed)
rgin_model.model.summary()




def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole?
            return True
        elif shell == 'TerminalInteractiveShell':  # Terminal running IPython?
            return False
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


img = None
if isnotebook():
    from keras.utils.vis_utils import model_to_dot
    from IPython.display import SVG
    img = SVG(model_to_dot(rgin_model.model, show_shapes=True).create(prog='dot', format='svg'))



#print("Evaluating RGIN ...", file=sys.stderr)
rgin = evaluate(RGIN())
print(rgin.groupby('Platform')['Platform', 'Speedup'].mean())


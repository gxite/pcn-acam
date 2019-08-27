import pickle as pkl

class Output():

    def __init__(self):
        self.payload = []
    
    def add(self,entry):
        self.payload.append(entry)

    def dump_pickle(self,out_pkl_path):
        '''output pickle path as argument'''
        pkl.dump(self.payload, open(out_pkl_path, 'wb'))
        

def load_pickle(pickle_path):
  """Loads pickle from given pickle path"""
  return pkl.load(open(pickle_path, 'rb'))
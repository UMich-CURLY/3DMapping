import numpy as np

idx_cat_dict = {0: 'bg', 1: 'bus', 2: 'ped', 3: 'other'}
cat_idx_dict = {'bg': 0, 'bus': 1, 'ped': 2, 'other': 3}


class Voxel:
    def __init__(self, idx_cat_map=idx_cat_dict, semantic_ind=0):

        self.idx_cat_map = idx_cat_map
        self.semantic_class_idx = semantic_ind
        self.semantic_odds = np.array([0]*len(idx_cat_map), dtype=np.int8)

        self.decay = -3
        self.accum = 1

    def set_class(self, next_sem_idx):
        """
        Increases probability of given semantic idx and decreases all other probablities
        until min or max is reached
        """
        ii8 = np.iinfo(np.uint8)

        prob_update = np.array([self.decay]*len(self.semantic_odds), dtype=np.int8)
        prob_update[next_sem_idx] = self.accum

        self.semantic_odds = np.clip(self.semantic_odds + prob_update,
                                ii8.min, ii8.max)   

        self.semantic_class_idx = np.argmax(self.semantic_odds)

    def get_class(self):
        return self.semantic_class_idx 

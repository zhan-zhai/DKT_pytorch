import numpy as np
from torch.utils.data import Dataset

class OtherDataset(Dataset):
    def __init__(self,group,n_skill,max_seq = 100,min_step=10):
        super(OtherDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = group
        self.min_step = min_step
        self.user_ids = []
        for user_id in group.index:
            q, qa = group.loc[user_id]

            if len(q) < 10:
                continue
            self.user_ids.append(user_id)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_ = self.samples.loc[index]
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        if seq_len >= self.max_seq:
            q[:] = q_[-self.max_seq:]
            qa[:] = qa_[-self.max_seq:]
        else:
            q[-seq_len:] = q_
            qa[-seq_len:] = qa_

        target_id = q[-1]
        label = qa[-1]

        q = q[:-1].astype(np.int)
        qa = qa[:-1].astype(np.int)
        x = q[:-1]
        x += (qa[:-1] == 1) * self.n_skill

        target_id = np.array([target_id]).astype(np.int)
        label = np.array([label]).astype(np.int)

        return x, target_id, label 

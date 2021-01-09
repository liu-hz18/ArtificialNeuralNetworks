import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from rnn_cell import RNNCell, GRUCell, LSTMCell

class RNN(nn.Module):
    def __init__(self,
            num_embed_units,  # pretrained wordvec size
            num_units,        # RNN units size
            num_layers,       # number of RNN layers
            cell,             # type of RNN Cell
            num_vocabs,       # vocabulary size
            wordvec,          # pretrained wordvec matrix
            dataloader):      # dataloader

        super().__init__()

        # load pretrained wordvec
        self.wordvec = wordvec
        # the dataloader
        self.dataloader = dataloader

        # TODO START
        # fill the parameter for multi-layer RNN
        cell_map = {
            'rnn': RNNCell,
            'lstm': LSTMCell,
            'gru': GRUCell,
        }
        if cell in cell_map:
            self.cells = nn.Sequential(\
                cell_map[cell](num_embed_units, num_units),
                *[cell_map[cell](num_units, num_units) for _ in range(num_layers - 1)]
            )
        else:
            raise NotImplementedError(f"cell {cell} is not supported.")
        self.num_layers = num_layers
        # TODO END

        # intialize other layers
        self.linear = nn.Linear(num_units, num_vocabs)

    def forward(self, batched_data, device):
        # Padded Sentences
        sent = torch.tensor(batched_data["sent"], dtype=torch.long, device=device) # shape: (batch_size, length)
        # An example:
        #   [
        #   [2, 4, 5, 6, 3, 0],   # first sentence: <go> how are you <eos> <pad>
        #   [2, 7, 3, 0, 0, 0],   # second sentence:  <go> hello <eos> <pad> <pad> <pad>
        #   [2, 7, 8, 1, 1, 3]    # third sentence: <go> hello i <unk> <unk> <eos>
        #   ]
        # You can use self.dataloader.convert_ids_to_sentence(sent[0]) to translate the first sentence to string in this batch.

        # Sentence Lengths
        length = torch.tensor(batched_data["sent_length"], dtype=torch.long, device=device) # shape: (batch)
        # An example (corresponding to the above 3 sentences):
        #   [5, 3, 6]

        batch_size, seqlen = sent.shape

        # TODO START
        # implement embedding layer
        embedding = F.embedding(sent[:, :-1], self.wordvec)
        embedding = F.dropout(embedding, p=0.2, training=self.training)
        # shape: (batch_size, length, num_embed_units)
        # TODO END

        now_state = []
        for cell in self.cells:
            now_state.append(cell.init(batch_size, device))

        loss = 0
        logits_per_step = []
        for i in range(seqlen - 1):
            hidden = embedding[:, i]
            for j, cell in enumerate(self.cells):
                hidden, now_state[j] = cell(hidden, now_state[j]) # shape: (batch_size, num_units)
                if j < self.num_layers-1:
                    hidden = F.dropout(hidden, 0.5, training=self.training)
            logits = self.linear(hidden)  # shape: (batch_size, num_vocabs)
            logits_per_step.append(logits)

        # TODO START
        # calculate loss
        logits = torch.stack(logits_per_step, dim=2)
        loss = F.cross_entropy(logits, sent[:, 1:], ignore_index=0)
        # TODO END

        return loss, torch.stack(logits_per_step, dim=1)

    def inference(self, batch_size, device, decode_strategy, temperature, max_probability):
        # First Tokens is <go>
        now_token = torch.tensor([self.dataloader.go_id] * batch_size, dtype=torch.long, device=device)
        flag = torch.tensor([1] * batch_size, dtype=torch.float, device=device)

        now_state = []
        for cell in self.cells:
            now_state.append(cell.init(batch_size, device))

        generated_tokens = []
        for _ in range(50): # max sentecne length

            # TODO START
            # translate now_token to embedding
            embedding = F.embedding(now_token, self.wordvec)
            embedding = F.dropout(embedding, p=0.2, training=self.training)
            # shape: (batch_size, num_embed_units)
            # TODO END

            hidden = embedding
            for j, cell in enumerate(self.cells):
                hidden, now_state[j] = cell(hidden, now_state[j])
                if j < self.num_layers-1:
                    hidden = F.dropout(hidden, 0.5, training=self.training)
            logits = self.linear(hidden) # shape: (batch_size, num_vocabs)

            if decode_strategy == "random":
                prob = (logits / temperature).softmax(dim=-1) # shape: (batch_size, num_vocabs)
                now_token = torch.multinomial(prob, 1)[:, 0] # shape: (batch_size)
            elif decode_strategy == "top-p":
                # TODO START
                # implement top-p samplings
                # shape: (batch_size, num_vocabs)
                prob = (logits / temperature).softmax(dim=-1)
                sort_probs, _ = torch.sort(prob, dim=-1, descending=False)
                sum_sort_probs = torch.cumsum(sort_probs, dim=-1)
                min_probs, _ = torch.max(torch.where(sum_sort_probs <= (1.0 - max_probability), sort_probs, torch.zeros_like(sort_probs)), dim=-1, keepdim=True)
                new_prob = torch.where(prob >= min_probs, prob, torch.zeros_like(prob))
                new_prob /= new_prob.sum(-1, keepdim=True)
                now_token = torch.multinomial(new_prob, 1)[:, 0]  # shape: (batch_size)
                # shape: (batch_size)
                # TODO END
            else:
                raise NotImplementedError("unknown decode strategy")

            generated_tokens.append(now_token)
            flag = flag * (now_token != self.dataloader.eos_id)

            if flag.sum().tolist() == 0: # all sequences has generated the <eos> token
                break

        return torch.stack(generated_tokens, dim=1).detach().cpu().numpy()

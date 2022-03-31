from typing import List
import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM


class Inputs:
    def __init__(self, tokenizer, texts: List[str]):
        self.tokenizer = tokenizer
        self.texts = texts
        self.tokens = [tokenizer.tokenize(txt) for txt in texts]
        self.inputs = tokenizer(texts, padding=True, return_tensors="pt")
        for k, v in self.inputs.items():
            setattr(self, k, v)

    def __getitem__(self, val):
        return Inputs(
            self.tokenizer, self.tokenizer.batch_decode(self.input_ids[val])
        )

    def __repr__(self):
        txts = repr(self.texts)
        txts = txts[:25] + ('...' if txts[:25] else '')
        return f'<{self.__class__.__name__}(texts={txts}, input_ids.shape={tuple(self.input_ids.shape)})>'


class Pred(torch.Tensor):
    pred = None

    @staticmethod
    def __new__(cls, pred, x, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)

    def __init__(self, pred, x):
        self.pred = pred

    def __getitem__(self, val):
        return self.__class__(self.pred, super().__getitem__(val))


class PredSequenceScores(Pred):
    def plot(self, log=False):
        self.pred.plot_sequence_scores(self, log)


class PredTokenScores(Pred):
    def for_tokens(self, token_ids):
        seq_scores = self.pred.token_vals(token_ids, self)
        return PredSequenceScores(self.pred, seq_scores)

    def for_inputs(self):
        return self.for_tokens(self.pred.inputs.input_ids)

    def plot(self):
        self.pred.plot_token_scores(self)


class Predictor:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    @classmethod
    def from_name(cls, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return cls(tokenizer, model)

    def __call__(self, texts):
        inputs = Inputs(self.tokenizer, texts)
        with torch.no_grad():
            logits = self.model(**inputs.inputs).logits
        return Prediction.from_inputs_with_logits(inputs, logits)


class Prediction:
    def __init__(self,
        inputs: Inputs,
        logits,
        loss_fct,
        loss,
        ppl,
        log_p,
        p,
    ):
        self.inputs = inputs
        self.tokenizer = self.inputs.tokenizer
        self.loss_fct = loss_fct
        self.logits = PredTokenScores(self, logits)
        self.loss = PredSequenceScores(self, loss)
        self.ppl = PredSequenceScores(self, ppl)
        self.log_p = PredTokenScores(self, log_p)
        self.p = PredTokenScores(self, p)

    @classmethod
    def from_inputs_with_logits(cls, inputs, logits):
        batch_size = logits.shape[0]
        with torch.no_grad():
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, logits.size(-1)), inputs.input_ids.view(-1)).view(batch_size, -1)
            ppl = torch.exp(loss)
            log_p = torch.nn.functional.log_softmax(logits, dim=-1)
            p = torch.exp(log_p)
            
        return cls(
            inputs,
            logits,
            loss_fct,
            loss,
            ppl,
            log_p,
            p,
        )

    @classmethod
    def from_text(cls, text):
        return cls.from_texts([text])

    def __repr__(self):
        return f'<{self.__class__.__name__}({self.inputs}, logits.shape={tuple(self.logits.shape)})>'

    def __len__(self):
        return len(self.inptus.input_ids)    

    @property
    def shape(self):
        return self.inptus.input_ids.shape

    def size(self, *args):
        return self.inptus.input_ids.size(*args)

    def __getitem__(self, val):
        if type(val) is int:
            return self.__getitem__(slice(val, val+1))
        return Prediction(
            self.inputs[val],
            self.logits[val],
            self.loss_fct,
            self.loss[val],
            self.ppl[val],
            self.log_p[val],
            self.p[val],
        )

    def token_vals(self, token_ids, scores):
        results = []
        for i, score_row in enumerate(scores):
            tkn_row = token_ids[i]
            results.append(
                score_row[range(tkn_row.shape[-1]), tkn_row]
            )
        return torch.stack(results)

    def input_token_vals(self, scores):
        return self.token_vals(self.inputs.input_ids, scores)

    def token_probs(self):
        return self.input_token_vals(self.p)

    def plot_sequence_scores(self, scores, log=False):
        # plot 1 score per token in each sequence
        assert len(scores.shape) == 2
        fig, ax = plt.subplots()
        n_seq, seq_len = scores.shape
        x_ticks = [[] for _ in range(seq_len)]
        for i, row in enumerate(scores):
            ax.bar(range(seq_len), row, alpha=0.5 if n_seq > 1 else 1.0)
            if log:
                ax.set_yscale('log')
            
            for i, tkn in enumerate(self.inputs.tokens[i]):
                if tkn not in x_ticks[i] and len(x_ticks[i]) < 3:
                    x_ticks[i].append(tkn)

        ax.set_xticks(range(seq_len), ['/'.join(ticks) for ticks in x_ticks], rotation=90)

    def plot_token_scores(self, token_scores, topk=5):
        # plot scores over all tokens
        fig, axs = plt.subplots(
            token_scores.shape[0] * token_scores.shape[1], figsize=(10, 2 * token_scores.shape[0] * token_scores.shape[1])
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        for batch_i in range(token_scores.shape[0]):
            for token_i in range(token_scores.shape[1]):
                ax = axs[token_scores.shape[1] * batch_i + token_i]

                hist, bin_edges = token_scores[batch_i, token_i].histogram()
                ax.bar(bin_edges[:-1], hist, width=bin_edges[:-1] - bin_edges[1:])
                ax.set_yscale('log')

                topk_tokens = []
                if topk:
                    vals, inds = token_scores[batch_i, token_i].topk(topk)
                    topk_tokens = self.tokenizer.batch_decode(inds)

                # show input sequence so far
                text_so_far = self.tokenizer.decode(self.inputs.input_ids[batch_i,:token_i])

                ax.text(
                    1.1, 0.95,
                    f'>>{text_so_far}\n' + '\n'.join(['- ' + repr(tk) for tk in topk_tokens]),
                    transform=ax.transAxes, fontsize=14, verticalalignment='top',
                    bbox=props
                )

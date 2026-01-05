import torch
from transformers import DynamicCache

class KVCacheManager:
    def __init__(
        self,
        mode="full",
        sink_size=4,
        window_size=256,
        h2o_heavy_size=256,
    ):
        assert mode in ["full", "streaming", "h2o"]
        self.mode = mode
        self.sink_size = sink_size
        self.window_size = window_size
        self.h2o_heavy_size = h2o_heavy_size

        self.past = None
        self.importance = None 
        
    def reset(self):
        """Resets the cache state."""
        self.past_key_values = None
        self.importance = None
        self.num_pruned = 0
        
    def update(self, past_key_values, attentions=None, **kwargs):
        if past_key_values is None:
            self.past = None
            self.importance = None
            return None

        if isinstance(past_key_values, DynamicCache):
            kv = past_key_values.to_legacy_cache()
        else:
            kv = past_key_values
            
        self.past = kv

        seq_len = kv[0][0].shape[2]

        # ---- FULL
        if self.mode == "full":
            self.past = kv
            self.past_key_values = DynamicCache.from_legacy_cache(kv)
            return self.past_key_values

        # ---- STREAMING
        if self.mode == "streaming":
            keep = self._streaming_indices(seq_len)
            self._prune_kv(keep)
            self.past_key_values = DynamicCache.from_legacy_cache(self.past)
            return self.past_key_values

        # ---- H2O
        if self.mode == "h2o":
            if attentions is not None:
                self._update_h2o_scores(attentions)

            keep = self._h2o_indices(seq_len)
            self._prune_kv(keep)
            self.past_key_values = DynamicCache.from_legacy_cache(self.past)
            return self.past_key_values


    def _streaming_indices(self, seq_len):
        if seq_len <= self.sink_size + self.window_size:
            return torch.arange(seq_len)

        sink = torch.arange(self.sink_size)
        window = torch.arange(seq_len - self.window_size, seq_len)
        return torch.cat([sink, window])

    def _h2o_indices(self, seq_len):
        if seq_len <= self.window_size + self.h2o_heavy_size:

            device = self.past[0][0].device if self.past else "cpu"
            return torch.arange(seq_len, device=device)

        device = self.past[0][0].device if self.past else "cpu"
        window = torch.arange(seq_len - self.window_size, seq_len, device=device)

        scores = self.importance
        if scores is None:
            return torch.arange(seq_len, device=device)
        
        candidates = scores[: seq_len - self.window_size]

        k = min(self.h2o_heavy_size, candidates.numel())
        _, heavy = torch.topk(candidates, k)
        
        keep = torch.cat([heavy, window])
        keep, _ = torch.sort(keep)
        return keep

    def _prune_kv(self, keep_idx):
        new_past = []
        for k, v in self.past:
            dim = k.shape[-1]
            idx = keep_idx.view(1, 1, -1, 1).expand(
                k.shape[0], k.shape[1], -1, dim
            )
            new_k = torch.gather(k, 2, idx)
            new_v = torch.gather(v, 2, idx)
            new_past.append((new_k, new_v))

        self.num_pruned += self.past[0][0].shape[2] - keep_idx.numel()
        self.past = tuple(new_past)

        if self.importance is not None:
            self.importance = self.importance[keep_idx]

    def _update_h2o_scores(self, attentions):

        new_scores = None
        for attn in attentions:
            attn_t = attn[0] if isinstance(attn, tuple) else attn
            score = attn_t.sum(dim=(0, 1, 2))
            
            if new_scores is None:
                new_scores = score
            else:
                new_scores = new_scores + score
        
        if new_scores is None:
            return
        
        if self.importance is None:
            self.importance = new_scores
        else:
            old_len = self.importance.shape[0]
            new_len = new_scores.shape[0]
            
            if new_len > old_len:
                diff = new_len - old_len
                new_scores[:old_len] = new_scores[:old_len] + self.importance
                self.importance = new_scores
            elif new_len == old_len:
                self.importance = self.importance + new_scores
            else:
                self.importance = self.importance[:new_len] + new_scores
    
    def get_cache_size(self):
        """Returns the number of tokens currently in the cache."""
        if self.past is None:
            return 0
        return self.past[0][0].shape[2]

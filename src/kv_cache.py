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

        self.past = None              # legacy kv
        # H2O specific: Store accumulated attention scores for each layer
        # List of tensors, one per layer: (batch, num_heads, seq_len)
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

        # ---- unwrap DynamicCache once
        if isinstance(past_key_values, DynamicCache):
            kv = past_key_values.to_legacy_cache()
        else:
            kv = past_key_values
            
        # Ensure self.past is set correctly for first run
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

    # --------------------------------------------------
    # index selection
    # --------------------------------------------------
    def _streaming_indices(self, seq_len):
        if seq_len <= self.sink_size + self.window_size:
            return torch.arange(seq_len)

        sink = torch.arange(self.sink_size)
        window = torch.arange(seq_len - self.window_size, seq_len)
        return torch.cat([sink, window])

    def _h2o_indices(self, seq_len):
        if seq_len <= self.window_size + self.h2o_heavy_size:
            # Must return tensor on correct device!
            # Use self.past to infer device if possible, or just CPU?
            # gather needs index on same device as k/v.
            device = self.past[0][0].device if self.past else "cpu"
            return torch.arange(seq_len, device=device)

        device = self.past[0][0].device if self.past else "cpu"
        window = torch.arange(seq_len - self.window_size, seq_len, device=device)

        # aggregate across layers
        if isinstance(self.importance, list) and len(self.importance) > 0:
            scores = torch.stack(self.importance).sum(0)
        else:
            scores = self.importance
        # scores should be on GPU if attentions were on GPU.
        
        candidates = scores[: seq_len - self.window_size]

        k = min(self.h2o_heavy_size, candidates.numel())
        _, heavy = torch.topk(candidates, k)
        
        # heavy indices will be on same device as scores
        
        keep = torch.cat([heavy, window])
        keep, _ = torch.sort(keep)
        return keep

    # --------------------------------------------------
    # pruning
    # --------------------------------------------------
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
            self.importance = [
                score[keep_idx] for score in self.importance
            ]

    # --------------------------------------------------
    # H2O score update
    # --------------------------------------------------
    def _update_h2o_scores(self, attentions):
        # attentions: tuple[layer] -> (B, H, q_len, k_len)
        if self.importance is None:
            self.importance = []

        for i, attn in enumerate(attentions):
            # attn: (B, H, q_len, k_len)
            # We want to sum over Batch, Heads, Query Length to get importance per Key token
            # This is heavy if done naively in Python loop
            # But here `attentions` is a tuple of tensors.
            
            # Optimization: 
            # `attn` is usually on GPU. `sum` is fast.
            # But we iterate over 32 layers in Python.
            # And we do `score.clone()`, `torch.cat`, `old + score`.
            # This is 32 * N kernel launches per step.
            # For small batch decoding, launch overhead is significant.
            
            attn_t = attn[0] if isinstance(attn, tuple) else attn
            
            # Sum over B, H, Q
            # score shape: (k_len)
            score = attn_t.sum(dim=(0, 1, 2)) 

            if i >= len(self.importance):
                self.importance.append(score)
            else:
                old = self.importance[i]
                # If seq_len increased (decoding), pad old scores
                # Usually k_len = old_len + 1
                diff = score.shape[0] - old.shape[0]
                if diff > 0:
                    # Pad old with zeros
                    pad = torch.zeros(diff, device=old.device, dtype=old.dtype)
                    old = torch.cat([old, pad])
                elif diff < 0:
                    # Should not happen in standard decoding unless we just pruned?
                    # If we pruned, `importance` was pruned too.
                    pass
                
                # In-place add if possible? 
                # old + score creates new tensor.
                self.importance[i] = old + score
    
    def get_cache_size(self):
        """Returns the number of tokens currently in the cache."""
        if self.past is None:
            return 0
        # self.past is tuple of tuples, first layer, first element (key), 3rd dim (seq_len)
        return self.past[0][0].shape[2]

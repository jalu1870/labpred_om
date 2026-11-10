"""Imputation and diagnosis models.

Both models share the same input encoding: each measurement day is a token whose features are the
(z-scored, imputed) values of all channels concatenated with a binary "observed" indicator per
channel, projected linearly to the hidden size. A learned sex embedding is added as an extra token
and the sequence is processed by a Qwen3 transformer backbone (rotary position embeddings over
calendar-day offsets).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3Model


def qwen3_backbone(hidden_size: int, num_heads: int, num_layers: int) -> Qwen3Model:
    config = Qwen3Config(
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        num_hidden_layers=num_layers,
        vocab_size=0,
        max_position_embeddings=512,
        hidden_act="gelu",
    )
    return Qwen3Model(config)


def encode_inputs(values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """[B, T, C] values (NaN allowed) + [B, T, C] observed flags -> [B, T, 2C]."""
    return torch.cat([values.nan_to_num(0.0), observed.float()], dim=-1)


class ImputationModel(nn.Module):
    """Reconstructs every channel of every measurement day from the full patient timeline.

    Token layout: [sex, day_1, ..., day_T]; position ids [0, pos_1 + 1, ..., pos_T + 1].
    """

    def __init__(self, n_channels: int = 40, hidden_size: int = 256, num_heads: int = 2, num_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(2 * n_channels, hidden_size)
        self.sex_embed = nn.Embedding(2, hidden_size)
        self.backbone = qwen3_backbone(hidden_size, num_heads, num_layers)
        self.output_proj = nn.Linear(hidden_size, n_channels)

    def forward(self, values, sex, attention_mask, position_ids):
        """values: [B, T, C] with NaN for missing entries. Returns reconstructions [B, T, C]."""
        B, T, _ = values.shape
        tokens = self.input_proj(encode_inputs(values, ~torch.isnan(values)))
        tokens = torch.cat([self.sex_embed(sex.long())[:, None], tokens], dim=1)
        pos = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=values.device), position_ids.long() + 1], dim=1)
        mask = torch.cat([torch.ones(B, 1, device=values.device), attention_mask.float()], dim=1)
        # The mask enters attention as an additive float bias (+1 for real days, 0 for padding),
        # exactly as during training of the released imputation weights.
        bias = mask[:, None, None, :].expand(B, 1, T + 1, T + 1).float()
        hidden = self.backbone(inputs_embeds=tokens, attention_mask=bias, position_ids=pos).last_hidden_state
        return self.output_proj(hidden)[:, 1:]

    @torch.no_grad()
    def impute(self, values, sex, attention_mask, position_ids):
        """Fill missing entries with model reconstructions; observed values are kept unchanged."""
        recon = self.forward(values, sex, attention_mask, position_ids).float()
        observed = ~torch.isnan(values)
        recon[observed] = values[observed]
        return recon


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden, mask):
        scores = self.attn(hidden).squeeze(-1).masked_fill(mask == 0, float("-inf"))
        return torch.bmm(F.softmax(scores, dim=1).unsqueeze(1), hidden).squeeze(1)


class DiagnosisModel(nn.Module):
    """Multi-label prediction of newly recorded diagnoses from one laboratory window.

    Token layout: [CLS, sex, day_1, ..., day_T]; position ids [0, 0, pos_1 + 1, ..., pos_T + 1].
    The sequence representation is an attention-pooled summary of all tokens, followed by a
    two-layer MLP with one logit per diagnosis.
    """

    def __init__(self, n_channels: int = 40, n_diagnoses: int = 162, hidden_size: int = 64,
                 num_heads: int = 2, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(2 * n_channels, hidden_size)
        self.sex_embed = nn.Embedding(2, hidden_size)
        self.backbone = qwen3_backbone(hidden_size, num_heads, num_layers)
        self.pool = AttentionPooling(hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, n_diagnoses)

    def forward(self, values, observed, sex, attention_mask, position_ids, bidirectional: bool = True):
        """values: [B, T+1, C] imputed inputs incl. a leading all-zero CLS row; observed: [B, T+1, C]
        flags of the original measurements; attention_mask/position_ids: [B, T+1] (CLS first)."""
        B = values.shape[0]
        tokens = self.input_proj(encode_inputs(values, observed))
        tokens = torch.cat([tokens[:, :1], self.sex_embed(sex.long())[:, None], tokens[:, 1:]], dim=1)
        zero = torch.zeros(B, 1, dtype=torch.long, device=values.device)
        pos = torch.cat([position_ids[:, :1].long(), zero, position_ids[:, 1:].long() + 1], dim=1)
        mask = torch.cat([attention_mask[:, :1], torch.ones(B, 1, device=values.device), attention_mask[:, 1:]], dim=1).float()
        attn = mask[:, None, None, :].expand(B, 1, mask.shape[1], mask.shape[1]).bool() if bidirectional else mask
        hidden = self.backbone(inputs_embeds=tokens, attention_mask=attn, position_ids=pos).last_hidden_state
        h = self.projection(self.pool(hidden, mask))
        return self.fc2(self.dropout(F.relu(self.fc1(h))))


# --- checkpoints -----------------------------------------------------------------------------------

_DIAGNOSIS_KEYS = [  # original training-code parameter names -> names used here
    ("seq_backbone.backbone.backbone.", "backbone."),
    ("seq_backbone.backbone.net.0.", "input_proj."),
    ("seq_backbone.backbone.sex_embed.", "sex_embed."),
    ("seq_backbone.pool.", "pool."),
    ("seq_backbone.projection.", "projection."),
]
_IMPUTATION_KEYS = [("net.0.", "input_proj.")]


def _rename(state, rules):
    out = {}
    for k, v in state.items():
        k = k.replace("_orig_mod.", "")
        for old, new in rules:
            if k.startswith(old):
                k = new + k[len(old):]
                break
        out[k] = v
    return out


def load_diagnosis_model(path, map_location="cpu", **kwargs) -> DiagnosisModel:
    state = _rename(torch.load(path, map_location=map_location, weights_only=True), _DIAGNOSIS_KEYS)
    state = {k: v for k, v in state.items() if not k.startswith("seq_backbone.backbone.age_embed")}
    model = DiagnosisModel(**kwargs)
    model.load_state_dict(state, strict=True)
    return model.eval()


def load_imputation_model(path, map_location="cpu", **kwargs) -> ImputationModel:
    model = ImputationModel(**kwargs)
    model.load_state_dict(_rename(torch.load(path, map_location=map_location, weights_only=True), _IMPUTATION_KEYS), strict=True)
    return model.eval()

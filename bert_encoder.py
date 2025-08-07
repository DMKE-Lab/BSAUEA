import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# ---------- 4.2.1 Input Preprocessing + BERT Encoding ----------
class NameBERTEncoder(nn.Module):
    def __init__(self, bert_model='bert-base-multilingual-cased', d_hidden=512, d_output=300):
        super(NameBERTEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert = BertModel.from_pretrained(bert_model)
        d_bert = self.bert.config.hidden_size  # typically 768

        # ---------- 4.2.2 MLP 映射层 ----------
        self.mlp = nn.Sequential(
            nn.Linear(d_bert, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_output)
        )

    def forward(self, entity_texts):
        """
        entity_texts: List[str], 每个实体的 name 或 description
        return: Tensor, shape (batch_size, d_output)
        """
        inputs = self.tokenizer(entity_texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.bert(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        projected = self.mlp(cls_embeddings)
        return projected  # shape: (batch_size, d_output)

# ---------- 4.2.3 Triplet Margin Loss ----------
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: torch.sum(torch.abs(x - y), dim=1),  # Manhattan distance
            margin=margin
        )

    def forward(self, anchor, positive, negative):
        """
        anchor, positive, negative: Tensor of shape (batch_size, d_output)
        return: scalar loss
        """
        return self.loss_fn(anchor, positive, negative)


if __name__ == "__main__":

    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = NameBERTEncoder().to(device)
    loss_fn = TripletLoss(margin=1.0)

    # 编码实体名称
    anchor_embed = encoder(sample_entities[""]).to(device)
    pos_embed = encoder(sample_entities[""]).to(device)
    neg_embed = encoder(sample_entities[""]).to(device)

    # 计算 Triplet Loss
    loss = loss_fn(anchor_embed, pos_embed, neg_embed)
    print("Triplet Loss:", loss.item())

"""
Baseline encoder-classifier model for CLARITY task.

Implements a transformer encoder (DeBERTa-v3-base) with classification head
for evasion detection (Subtask 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig
)
from typing import Dict, Optional


# Evasion to Clarity group mapping (for hierarchical regularization)
# Group 0: Clear Reply = [Explicit]
# Group 1: Ambivalent = [Implicit, Dodging, General, Deflection, Partial/half-answer]
# Group 2: Clear Non-Reply = [Declining to answer, Claims ignorance, Clarification]
EVASION_TO_CLARITY_GROUP = [0, 1, 1, 1, 1, 1, 2, 2, 2]  # Indices match EVASION_LABELS order


class HierarchicalLabelSmoothing(nn.Module):
    """
    Hierarchical label smoothing that distributes smoothing mass
    preferentially within the same clarity group.
    
    Rationale: Annotator disagreement tends to stay within a clarity group,
    so we smooth more within-group than across-group.
    """
    
    def __init__(self, smoothing: float = 0.1, within_group_ratio: float = 0.7):
        """
        Args:
            smoothing: Total smoothing mass (epsilon)
            within_group_ratio: Fraction of smoothing to keep within same group
        """
        super().__init__()
        self.smoothing = smoothing
        self.within_group_ratio = within_group_ratio
        
        # Precompute smoothing distributions per class
        self.register_buffer('smooth_dist', self._compute_smooth_distributions())
    
    def _compute_smooth_distributions(self) -> torch.Tensor:
        """Compute smoothing distribution for each target class."""
        n_classes = 9
        groups = torch.tensor(EVASION_TO_CLARITY_GROUP)
        
        distributions = []
        for target in range(n_classes):
            target_group = groups[target]
            same_group_mask = (groups == target_group).float()
            diff_group_mask = (groups != target_group).float()
            
            # Count classes in each category (excluding target)
            n_same = same_group_mask.sum() - 1  # Exclude target
            n_diff = diff_group_mask.sum()
            
            # Distribute smoothing
            within_mass = self.smoothing * self.within_group_ratio
            across_mass = self.smoothing * (1 - self.within_group_ratio)
            
            dist = torch.zeros(n_classes)
            dist[target] = 1.0 - self.smoothing  # Main probability
            
            # Distribute within-group smoothing
            if n_same > 0:
                for i in range(n_classes):
                    if i != target and groups[i] == target_group:
                        dist[i] = within_mass / n_same
            
            # Distribute across-group smoothing
            if n_diff > 0:
                for i in range(n_classes):
                    if groups[i] != target_group:
                        dist[i] = across_mass / n_diff
            
            distributions.append(dist)
        
        return torch.stack(distributions)  # [n_classes, n_classes]
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy with hierarchical label smoothing.
        
        Args:
            logits: [batch_size, n_classes]
            targets: [batch_size]
        
        Returns:
            Loss scalar
        """
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get smoothed target distributions
        if self.smooth_dist.device != targets.device:
            self.smooth_dist = self.smooth_dist.to(targets.device)
        
        target_dist = self.smooth_dist[targets]  # [batch_size, n_classes]
        
        # Compute cross-entropy with smoothed targets
        loss = -(target_dist * log_probs).sum(dim=-1).mean()
        
        return loss


class HierarchicalRegularizer(nn.Module):
    """
    Adds hierarchical regularization by computing clarity-level loss
    from aggregated evasion logits.
    
    This enforces the taxonomy constraint (evasion -> clarity mapping)
    without adding a second classification head.
    """
    
    def __init__(self, lambda_clarity: float = 0.3):
        """
        Args:
            lambda_clarity: Weight for clarity regularization term
        """
        super().__init__()
        self.lambda_clarity = lambda_clarity
        
        # Evasion indices per clarity group
        # Clear Reply: Explicit (0)
        # Ambivalent: Implicit (1), Dodging (2), General (3), Deflection (4), Partial (5)
        # Clear Non-Reply: Declining (6), Claims ignorance (7), Clarification (8)
        self.clarity_groups = [
            [0],           # Clear Reply
            [1, 2, 3, 4, 5],  # Ambivalent
            [6, 7, 8]      # Clear Non-Reply
        ]
    
    def compute_clarity_logits(self, evasion_logits: torch.Tensor) -> torch.Tensor:
        """
        Aggregate evasion logits into clarity logits using log-sum-exp.
        
        Args:
            evasion_logits: [batch_size, 9]
        
        Returns:
            clarity_logits: [batch_size, 3]
        """
        clarity_logits = []
        for group_indices in self.clarity_groups:
            group_logits = evasion_logits[:, group_indices]
            # Log-sum-exp aggregation
            group_clarity = torch.logsumexp(group_logits, dim=1)
            clarity_logits.append(group_clarity)
        
        return torch.stack(clarity_logits, dim=1)  # [batch_size, 3]
    
    def forward(
        self, 
        evasion_logits: torch.Tensor, 
        evasion_labels: torch.Tensor,
        evasion_loss: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined evasion + clarity loss.
        
        Args:
            evasion_logits: [batch_size, 9]
            evasion_labels: [batch_size] (evasion label indices)
            evasion_loss: Already computed evasion loss
            class_weights: Optional class weights (not used for clarity)
        
        Returns:
            Combined loss = evasion_loss + lambda * clarity_loss
        """
        # Derive clarity labels from evasion labels
        groups = torch.tensor(EVASION_TO_CLARITY_GROUP, device=evasion_labels.device)
        clarity_labels = groups[evasion_labels]
        
        # Compute clarity logits
        clarity_logits = self.compute_clarity_logits(evasion_logits)
        
        # Clarity loss (no class weights, just regularization)
        clarity_loss = F.cross_entropy(clarity_logits, clarity_labels)
        
        # Combined loss
        return evasion_loss + self.lambda_clarity * clarity_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the true class.
    
    This loss down-weights easy examples and focuses on hard examples,
    which is particularly useful for imbalanced datasets.
    """
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        """
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Class weighting (optional, can be used with class_weights)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Model logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        
        Returns:
            Focal loss scalar
        """
        # Compute cross-entropy loss (without reduction)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Compute p_t (probability of true class)
        p_t = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Reduce
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EvasionClassifier(nn.Module):
    """
    Transformer encoder + classification head for evasion detection.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 9,
        dropout: float = 0.1,
        pooling: str = "cls",
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        use_hierarchical_reg: bool = False,
        lambda_clarity: float = 0.3,
        use_hierarchical_smoothing: bool = False,
        smoothing: float = 0.1,
        within_group_ratio: float = 0.7,
        label_smoothing: float = 0.0
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            num_labels: Number of output classes
            dropout: Dropout probability
            pooling: Pooling strategy ('cls' or 'mean')
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.pooling = pooling
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.use_hierarchical_reg = use_hierarchical_reg
        self.use_hierarchical_smoothing = use_hierarchical_smoothing
        self.label_smoothing = label_smoothing
        
        # Load pretrained encoder
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Classification head
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Initialize classifier weights
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.classifier.bias.data.zero_()
        
        # Initialize hierarchical components
        self.hierarchical_reg = None
        self.hierarchical_smoothing = None
        
        if use_hierarchical_reg:
            self.hierarchical_reg = HierarchicalRegularizer(lambda_clarity=lambda_clarity)
        
        if use_hierarchical_smoothing:
            self.hierarchical_smoothing = HierarchicalLabelSmoothing(
                smoothing=smoothing,
                within_group_ratio=within_group_ratio
            )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs (optional)
            labels: Ground truth labels [batch_size] (optional)
            class_weights: Class weights for loss [num_classes] (optional)
        
        Returns:
            Dict with:
                - logits: Class logits [batch_size, num_labels]
                - loss: Cross-entropy loss (if labels provided)
                - hidden_states: Pooled hidden states [batch_size, hidden_size]
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Pool
        if self.pooling == "cls":
            pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        elif self.pooling == "mean":
            # Mean pooling over non-padding tokens
            hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            sum_hidden = torch.sum(hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        result = {
            "logits": logits,
            "hidden_states": pooled
        }
        
        # Compute loss if labels provided
        if labels is not None:
            # Choose loss function based on configuration
            if self.use_hierarchical_smoothing and self.hierarchical_smoothing is not None:
                # Hierarchical label smoothing (takes precedence over other losses)
                loss = self.hierarchical_smoothing(logits, labels)
            elif self.use_focal_loss:
                # Focal loss (with optional alpha=class_weights)
                focal_alpha = class_weights if class_weights is not None else None
                criterion = FocalLoss(gamma=self.focal_gamma, alpha=focal_alpha)
                loss = criterion(logits, labels)
            elif class_weights is not None:
                # Weighted cross-entropy with optional label smoothing
                criterion = nn.CrossEntropyLoss(
                    weight=class_weights, 
                    label_smoothing=self.label_smoothing
                )
                loss = criterion(logits, labels)
            else:
                # Plain cross-entropy with optional label smoothing
                criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
                loss = criterion(logits, labels)
            
            # Add hierarchical regularization if enabled
            if self.use_hierarchical_reg and self.hierarchical_reg is not None:
                loss = self.hierarchical_reg(logits, labels, loss, class_weights)
            
            result["loss"] = loss
        
        return result
    
    def get_tokenizer(self):
        """Get the tokenizer for this model."""
        return AutoTokenizer.from_pretrained(self.model_name)


def load_model_and_tokenizer(
    model_name: str = "microsoft/deberta-v3-base",
    num_labels: int = 9,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    use_hierarchical_reg: bool = False,
    lambda_clarity: float = 0.3,
    use_hierarchical_smoothing: bool = False,
    smoothing: float = 0.1,
    within_group_ratio: float = 0.7,
    label_smoothing: float = 0.0
):
    """
    Load model and tokenizer.
    
    Args:
        model_name: HuggingFace model identifier
        num_labels: Number of classes
        device: Device to load model on
        use_focal_loss: Whether to use focal loss
        focal_gamma: Focal loss gamma parameter
        use_hierarchical_reg: Whether to use hierarchical regularization
        lambda_clarity: Weight for clarity regularization
        use_hierarchical_smoothing: Whether to use hierarchical label smoothing
        smoothing: Label smoothing epsilon
        within_group_ratio: Fraction of smoothing within same clarity group
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model = EvasionClassifier(
        model_name=model_name,
        num_labels=num_labels,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        use_hierarchical_reg=use_hierarchical_reg,
        lambda_clarity=lambda_clarity,
        use_hierarchical_smoothing=use_hierarchical_smoothing,
        smoothing=smoothing,
        within_group_ratio=within_group_ratio,
        label_smoothing=label_smoothing
    )
    model = model.to(device)
    
    tokenizer = model.get_tokenizer()
    
    return model, tokenizer


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("Creating model...")
    model, tokenizer = load_model_and_tokenizer(
        model_name="microsoft/deberta-v3-base",
        num_labels=9,
        device="cpu"
    )
    
    print(f"Model: {model.model_name}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    text = "Question: How would you respond? Answer: Well, I think..."
    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Logits: {outputs['logits'][0][:3].tolist()}")  # First 3 classes
    
    # Test with labels
    labels = torch.tensor([0])
    outputs = model(**inputs, labels=labels)
    print(f"Loss: {outputs['loss'].item():.4f}")

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.prompt import HiVePrompt


class SoftPromptModel(nn.Module):
    """
    Wrapper around a frozen causal LM with a learnable HiVePrompt prefix.

    The base model is fully frozen; only soft_prompt parameters are trained.
    """

    def __init__(
        self,
        model_name: str,
        soft_prompt_length: int,
        low_rank_dim: int,
        device: str,
    ):
        super().__init__()
        self.device = device

        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.base_model.resize_token_embeddings(len(self.tokenizer))
            pad_id = self.tokenizer.pad_token_id
            with torch.no_grad():
                self.base_model.get_input_embeddings().weight[pad_id].zero_()
            self.base_model.config.pad_token_id = pad_id

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        print(self.tokenizer.padding_side)

        self.soft_prompt        = None
        self.hidden_dim         = self.base_model.config.hidden_size
        self.soft_prompt_length = soft_prompt_length
        self.low_rank_dim       = low_rank_dim

    def train(self, mode: bool = True):
        """Keep base model in eval mode at all times."""
        super().train(mode)
        self.base_model.eval()
        return self

    def init_soft_prompt(
        self,
        num_tasks: int,
        soft_prompt_template: list[str],
        task_embeddings: torch.Tensor,
    ):
        """Initialize the HiVePrompt module and attach it to the model."""
        self.soft_prompt = HiVePrompt(
            base_model=self.base_model,
            tokenizer=self.tokenizer,
            soft_prompt_template=soft_prompt_template,
            task_embeddings=task_embeddings,
            num_tasks=num_tasks,
            soft_prompt_length=self.soft_prompt_length,
            low_rank_dim=self.low_rank_dim,
            hidden_dim=self.hidden_dim,
            device=self.device,
        ).to(self.device)

    def forward(
        self,
        ids: list[str],
        routing_input_ids: torch.Tensor,
        routing_attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        task_ids: torch.Tensor = None,
    ):
        """
        Forward pass with soft prompt prepended.

        1. Compute routing embeddings for the prompt module
        2. Get soft prompt from HiVePrompt
        3. Prepend prompt to input embeddings
        4. Extend attention mask and labels accordingly
        5. Run base model forward
        """
        embed_layer = self.base_model.get_input_embeddings()

        # 1. Routing embeddings
        routing_embeds = embed_layer(routing_input_ids)

        # 2. Soft prompt
        prompt_out     = self.soft_prompt(
            ids=ids,
            input_embeds=routing_embeds,
            attention_mask=routing_attention_mask,
            task_ids=task_ids,
        )
        batched_prompt = prompt_out['batched_prompt']

        # 3. Input embeddings
        input_embeds   = embed_layer(input_ids)
        inputs_embeds  = torch.cat([batched_prompt, input_embeds], dim=1)

        # 4. Extend attention mask and labels
        B           = input_ids.shape[0]
        prompt_mask = torch.ones(B, self.soft_prompt.soft_prompt_length, device=self.device)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        prompt_labels = torch.full((B, self.soft_prompt.soft_prompt_length), -100, device=self.device)
        labels        = torch.cat([prompt_labels, labels], dim=1)

        # 5. Base model forward
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs, prompt_out

    def generate(
        self,
        ids: list[str],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_ids: torch.Tensor,
        **kwargs,
    ):
        """Generate with soft prompt prepended."""
        embed_layer  = self.base_model.get_input_embeddings()
        input_embeds = embed_layer(input_ids)

        prompt_out     = self.soft_prompt(
            ids=ids,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            task_ids=task_ids,
        )
        batched_prompt = prompt_out['batched_prompt']

        inputs_embeds  = torch.cat([batched_prompt, input_embeds], dim=1)

        B           = input_ids.shape[0]
        prompt_mask = torch.ones(B, self.soft_prompt.soft_prompt_length, device=self.device)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        return self.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            eos_token_id=self.base_model.config.eos_token_id,
            pad_token_id=self.base_model.config.pad_token_id,
            **kwargs,
        )

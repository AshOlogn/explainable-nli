import torch
from torch import nn
from typing import Optional, Tuple
from transformers import BartPretrainedModel, BartConfig, BartModel
from transformers.models.bart.modeling_bart import BartClassificationHead
from transformers.modeling_outputs import ModelOutput

class ExplanatoryNLIOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    classification_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class BartForExplanatoryNLI(BartPretrainedModel):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)

        # used for classification
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

        # used for explanation generation
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        classification_labels=None,
        explanation_labels=None,
        alpha=0.5,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        classification_labels: (batch_size,)
        explanation_labels: (batch_size, seq_length)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if classification_labels is not None or explanation_labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        if explanation_labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(explanation_labels, self.config.pad_token_id, self.config.decoder_start_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # (batch_size, seq_length, hidden_size)
        encoder_hidden_states = outputs.encoder_last_hidden_state
        decoder_hidden_states = outputs.last_hidden_state

        ##########################
        ###   Classification   ###
        ##########################

        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = encoder_hidden_states[eos_mask, :].view(encoder_hidden_states.size(0), -1, encoder_hidden_states.size(-1))[:,-1,:]
        classification_logits = self.classification_head(sentence_representation)

        classification_loss = None
        if classification_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(classification_logits.view(-1, self.config.num_labels), classification_labels.view(-1))

        ##########################
        ###     Generation     ###
        ##########################

        lm_logits = self.lm_head(decoder_hidden_states) + self.final_logits_bias

        explanation_loss = None
        if explanation_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            explanation_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), explanation_labels.view(-1))

        loss = None
        if classification_loss is not None and explanation_loss is not None:
            loss = alpha * classification_loss + (1-alpha) * explanation_loss
        elif classification_loss is not None:
            loss = classification_loss

        if not return_dict:
            output = (classification_logits, lm_logits) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ExplanatoryNLIOutput(
            loss=loss,
            logits=lm_logits,
            classification_logits=classification_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
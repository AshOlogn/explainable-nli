from dataclasses import dataclass
import torch
from torch import nn
from typing import Optional, Tuple
from transformers import BartPretrainedModel, BartConfig, BartModel
from transformers.models.bart.modeling_bart import BartClassificationHead
from transformers.modeling_outputs import ModelOutput

@dataclass
class ExplanatoryNLIOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    classification_logits: torch.FloatTensor = None

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
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)

        self.reduce = kwargs['reduce']
        self.alpha = kwargs['alpha']

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

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

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
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        classification_labels=None,
        explanation_labels=None,
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
        if not return_dict:
            raise NotImplementedError("Haven't double-checked output order if return_dict=False, currently not supported!")

        if classification_labels is not None or explanation_labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        if explanation_labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    explanation_labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
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

        classification_loss = None
        classification_logits = None

        if input_ids is not None:

            if self.reduce == 'eos':
                eos_mask = input_ids.eq(self.config.eos_token_id)
                if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                    raise ValueError("All examples must have the same number of <eos> tokens.")
                sentence_representation = encoder_hidden_states[eos_mask, :].view(encoder_hidden_states.size(0), -1, encoder_hidden_states.size(-1))[:,-1,:]

            elif self.reduce == 'mean':
                # (batch_size, seq_length)
                attention_mask

                # (batch_size, seq_length, hidden_size)
                encoder_hidden_states

                # (batch_size,)
                input_lengths = attention_mask.sum(dim=1)

                # (batch_size, hidden_size)
                mean_encoder_hidden_states = (encoder_hidden_states * attention_mask.unsqueeze(-1).expand_as(encoder_hidden_states)).sum(dim=1)
                mean_encoder_hidden_states /= input_lengths.unsqueeze(-1)
                sentence_representation = mean_encoder_hidden_states
                print(sentence_representation.shape)

            classification_logits = self.classification_head(sentence_representation)

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
            loss = self.alpha * classification_loss + (1-self.alpha) * explanation_loss
        elif classification_loss is not None:
            loss = classification_loss

        return ExplanatoryNLIOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            classification_logits=classification_logits,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

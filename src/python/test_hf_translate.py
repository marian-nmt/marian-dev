"""Using OPUS original model. Requires:
1. Rename the vocab file to vocab.json
2. add a <pad> token to vocab
"""
import os
from typing import Optional, Union, List
import torch

from pymarian import Translator
from transformers import PretrainedConfig, PreTrainedModel, MarianTokenizer


class PyMarianConfig(PretrainedConfig):
    r"""
    ```"""
    model_type = "pymarian"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size=50265,
        decoder_vocab_size=None,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        decoder_start_token_id=58100,
        classifier_dropout=0.0,
        scale_embedding=False,
        pad_token_id=58100,
        eos_token_id=0,
        forced_eos_token_id=0,
        share_encoder_decoder_embeddings=True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.decoder_vocab_size = decoder_vocab_size or vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.share_encoder_decoder_embeddings = share_encoder_decoder_embeddings
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )


class PyMarianTokenizer(MarianTokenizer):
    def _convert_token_to_id(self, token):
        return token

class PyMarianModel(PreTrainedModel):
    config_class = PyMarianConfig
    base_model_prefix = "pymarian"

    def __init__(self, config: PyMarianConfig):
        super().__init__(config)
        self.marian_model = None

    @classmethod
    def from_pretrained(cls, model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        """
        Load a pretrained model and return a PyMarianModel.
        """
        config = kwargs.pop("config", None)

        # model = super().from_pretrained(model_name_or_path, *model_args, **kwargs)
        model = cls(config)

        # marian_configs = _convert_hf_config_to_marian_config(config)
        model.marian_model = Translator(f"--config {model_name_or_path}/decoder.yml")
        return model

    def forward(
        self,
        input_tokens: List[str]
    ) -> List[str]:
        input_tokens = [' '.join(x) for x in input_tokens]
        return self.marian_model.translate(input_tokens)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> List[str]:

        input_tokens = [' '.join(x) for x in inputs]
        return self.marian_model.translate(input_tokens, **kwargs)


def main():
    model_path = '/home/alferre/models/hf-opus-2020-02-26'

    config = PyMarianConfig()
    tokenizer = PyMarianTokenizer.from_pretrained(model_path)
    model = PyMarianModel.from_pretrained(model_path, config=config)

    src_text = [
        "What is my translation?",
        "Hello World!",
        "I am a translator, but this is a much longer sequence than I thought",
        "Another supposition is that the author of the Liber Pontificals gives the papal interpretation of a grant that had been expressed by Pippin in ambiguous terms; and this view is supported by the history of the subsequent controversy between king and pope."
    ]

    batch_tokens = [tokenizer._tokenize(x) for x in src_text]

    translated_tokens = model.generate(inputs=batch_tokens, beam_size=1)
    print(translated_tokens)

    translated_tokens = model.generate(inputs=batch_tokens, beam_size=10)
    print(translated_tokens)


if __name__ == '__main__':
    main()
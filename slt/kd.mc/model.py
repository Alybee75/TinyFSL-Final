# coding: utf-8
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from itertools import groupby
from kd.mc.initialization import initialize_model
from kd.mc.embeddings import Embeddings, SpatialEmbeddings
from kd.mc.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from kd.mc.decoders import Decoder, RecurrentDecoder, TransformerDecoder
from kd.mc.search import beam_search, greedy
from torchtext.data import Field, RawField
from kd.mc.vocabulary import (
    TextVocabulary,
    GlossVocabulary,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
)
from kd.mc.batch import Batch
from kd.mc.helpers import freeze_params, load_config, load_checkpoint
from torch import Tensor
from typing import Union
import torch

class SignModel(nn.Module):
    """
    Base Model class
    """

    def __init__(
        self,
        encoder: Encoder,
        gloss_output_layer: nn.Module,
        decoder: Decoder,
        sgn_embed: SpatialEmbeddings,
        txt_embed: Embeddings,
        gls_vocab: GlossVocabulary,
        txt_vocab: TextVocabulary,
        do_recognition: bool = True,
        do_translation: bool = True,
    ):
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param sgn_embed: spatial feature frame embeddings
        :param txt_embed: spoken language word embedding
        :param gls_vocab: gls vocabulary
        :param txt_vocab: spoken language vocabulary
        :param do_recognition: flag to build the model with recognition output.
        :param do_translation: flag to build the model with translation decoder.
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.sgn_embed = sgn_embed
        self.txt_embed = txt_embed

        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab

        self.txt_bos_index = self.txt_vocab.stoi[BOS_TOKEN]
        self.txt_pad_index = self.txt_vocab.stoi[PAD_TOKEN]
        self.txt_eos_index = self.txt_vocab.stoi[EOS_TOKEN]

        self.gloss_output_layer = gloss_output_layer
        self.do_recognition = do_recognition
        self.do_translation = do_translation

    # pylint: disable=arguments-differ
    def forward(
        self,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        txt_input: Tensor,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param sgn: source input
        :param sgn_mask: source mask
        :param sgn_lengths: length of source inputs
        :param txt_input: target input
        :param txt_mask: target mask
        :return: decoder outputs
        """
        encoder_output, encoder_hidden = self.encode(
            sgn=sgn, sgn_mask=sgn_mask, sgn_length=sgn_lengths
        )

        if self.do_recognition:
            # Gloss Recognition Part
            # N x T x C
            gloss_scores = self.gloss_output_layer(encoder_output)
            # N x T x C
            gloss_probabilities = gloss_scores.log_softmax(2)
            # Turn it into T x N x C
            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
        else:
            gloss_probabilities = None

        if self.do_translation:
            unroll_steps = txt_input.size(1)
            decoder_outputs = self.decode(
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                sgn_mask=sgn_mask,
                txt_input=txt_input,
                unroll_steps=unroll_steps,
                txt_mask=txt_mask,
            )
        else:
            decoder_outputs = None

        return decoder_outputs, gloss_probabilities

    def encode(
        self, sgn: Tensor, sgn_mask: Tensor, sgn_length: Tensor
    ) -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param sgn:
        :param sgn_mask:
        :param sgn_length:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder(
            embed_src=self.sgn_embed(x=sgn, mask=sgn_mask),
            src_length=sgn_length,
            mask=sgn_mask,
        )

    def decode(
        self,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        sgn_mask: Tensor,
        txt_input: Tensor,
        unroll_steps: int,
        decoder_hidden: Tensor = None,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param sgn_mask: sign sequence mask, 1 at valid tokens
        :param txt_input: spoken language sentence inputs
        :param unroll_steps: number of steps to unroll the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param txt_mask: mask for spoken language words
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=sgn_mask,
            trg_embed=self.txt_embed(x=txt_input, mask=txt_mask),
            trg_mask=txt_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
        )

    def get_loss_for_batch(
        self,
        batch: Batch,
        isSoftPred: bool,
        teacher_model,  # Add teacher model as a parameter
        T: float,  # Temperature for distillation
        alpha: float,  # Weighting factor between traditional and distillation loss
        recognition_loss_function: nn.Module,
        translation_loss_function: nn.Module,
        recognition_loss_weight: float,
        translation_loss_weight: float,
    ) -> (Tensor, Tensor):
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param recognition_loss_function: Sign Language Recognition Loss Function (CTC)
        :param translation_loss_function: Sign Language Translation Loss Function (XEntropy)
        :param recognition_loss_weight: Weight for recognition loss
        :param translation_loss_weight: Weight for translation loss
        :return: recognition_loss: sum of losses over sequences in the batch
        :return: translation_loss: sum of losses over non-pad elements in the batch
        """
        # pylint: disable=unused-variable

        # Student model forward pass
        student_decoder_outputs, student_gloss_probabilities = self.forward(
            sgn=batch.sgn,
            sgn_mask=batch.sgn_mask,
            sgn_lengths=batch.sgn_lengths,
            txt_input=batch.txt_input,
            txt_mask=batch.txt_mask,
        )
        
        # Teacher model forward pass
        with torch.no_grad():
            teacher_decoder_outputs, teacher_gloss_probabilities = teacher_model.forward(
                sgn=batch.sgn,
                sgn_mask=batch.sgn_mask,
                sgn_lengths=batch.sgn_lengths,
                txt_input=batch.txt_input,
                txt_mask=batch.txt_mask,
            )

        # Initialize losses
        recognition_loss, translation_loss = None, None
        recognition_distill_loss, translation_distill_loss = None, None

        if isSoftPred:
            if self.do_recognition:
                assert student_gloss_probabilities is not None and teacher_gloss_probabilities is not None
                # Calculate Recognition Loss
                recognition_loss = recognition_loss_function(
                    student_gloss_probabilities,
                    batch.gls,
                    batch.sgn_lengths.long(),
                    batch.gls_lengths.long(),
                ) * recognition_loss_weight
    
                recognition_distill_loss = distillation_loss(
                    student_gloss_probabilities, teacher_gloss_probabilities, T, alpha
                ) if self.do_recognition else 0
    
            if self.do_translation:
                assert student_decoder_outputs is not None and teacher_decoder_outputs is not None
                student_word_outputs, _, _, _ = student_decoder_outputs
                teacher_word_outputs, _, _, _ = teacher_decoder_outputs
                
                # Calculate Translation Loss
                student_txt_log_probs = F.log_softmax(student_word_outputs, dim=-1)
                translation_loss = translation_loss_function(student_txt_log_probs, batch.txt) * translation_loss_weight
    
                # Calculate Translation Distillation Loss
                # translation_distill_loss = distillation_loss(
                #     student_word_outputs, teacher_word_outputs[:, :, 1:], T, alpha
                # ) if self.do_translation else 0

                translation_distill_loss = distillation_loss(
                    student_word_outputs, teacher_word_outputs, T, alpha
                ) if self.do_translation else 0
    
            # Combine traditional loss and distillation loss
            combined_recognition_loss = None
            combined_translation_loss = None
            distillation_scaling = (1 - alpha) * (T ** 2)

            # Adjusted combined recognition loss calculation
            if recognition_loss is not None:
                combined_recognition_loss = alpha * recognition_loss + distillation_scaling * recognition_distill_loss
            if translation_loss is not None:
                combined_translation_loss = alpha * translation_loss + distillation_scaling * translation_distill_loss
    
            return combined_recognition_loss, combined_translation_loss
            
        else:
            if self.do_recognition:
                assert student_gloss_probabilities is not None
                # Calculate Recognition Loss
                recognition_loss = (
                    recognition_loss_function(
                        student_gloss_probabilities,
                        batch.gls,
                        batch.sgn_lengths.long(),
                        batch.gls_lengths.long(),
                    )
                    * recognition_loss_weight
                )
            else:
                recognition_loss = None
    
            if self.do_translation:
                assert student_decoder_outputs is not None
                student_word_outputs, _, _, _ = student_decoder_outputs
                # Calculate Translation Loss
                txt_log_probs = F.log_softmax(student_word_outputs, dim=-1)
                translation_loss = (
                    translation_loss_function(txt_log_probs, batch.txt)
                    * translation_loss_weight
                )
            else:
                translation_loss = None
    
            return recognition_loss, translation_loss
            
    def run_batch(
        self,
        batch: Batch,
        recognition_beam_size: int = 1,
        translation_beam_size: int = 1,
        translation_beam_alpha: float = -1,
        translation_max_output_length: int = 100,
    ) -> (np.array, np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param recognition_beam_size: size of the beam for CTC beam search
            if 1 use greedy
        :param translation_beam_size: size of the beam for translation beam search
            if 1 use greedy
        :param translation_beam_alpha: alpha value for beam search
        :param translation_max_output_length: maximum length of translation hypotheses
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """

        encoder_output, encoder_hidden = self.encode(
            sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths
        )

        if self.do_recognition:
            # Gloss Recognition Part
            # N x T x C
            gloss_scores = self.gloss_output_layer(encoder_output)
            # N x T x C
            gloss_probabilities = gloss_scores.log_softmax(2)
            # Turn it into T x N x C
            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
            gloss_probabilities = gloss_probabilities.cpu().detach().numpy()
            tf_gloss_probabilities = np.concatenate(
                (gloss_probabilities[:, :, 1:], gloss_probabilities[:, :, 0, None]),
                axis=-1,
            )

            assert recognition_beam_size > 0
            ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
                inputs=tf_gloss_probabilities,
                sequence_length=batch.sgn_lengths.cpu().detach().numpy(),
                beam_width=recognition_beam_size,
                top_paths=1,
            )
            ctc_decode = ctc_decode[0]
            # Create a decoded gloss list for each sample
            tmp_gloss_sequences = [[] for i in range(gloss_scores.shape[0])]
            for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
                tmp_gloss_sequences[dense_idx[0]].append(
                    ctc_decode.values[value_idx].numpy() + 1
                )
            decoded_gloss_sequences = []
            for seq_idx in range(0, len(tmp_gloss_sequences)):
                decoded_gloss_sequences.append(
                    [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
                )
        else:
            decoded_gloss_sequences = None

        if self.do_translation:
            # greedy decoding
            if translation_beam_size < 2:
                stacked_txt_output, stacked_attention_scores = greedy(
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=batch.sgn_mask,
                    embed=self.txt_embed,
                    bos_index=self.txt_bos_index,
                    eos_index=self.txt_eos_index,
                    decoder=self.decoder,
                    max_output_length=translation_max_output_length,
                )
                # batch, time, max_sgn_length
            else:  # beam size
                stacked_txt_output, stacked_attention_scores = beam_search(
                    size=translation_beam_size,
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=batch.sgn_mask,
                    embed=self.txt_embed,
                    max_output_length=translation_max_output_length,
                    alpha=translation_beam_alpha,
                    eos_index=self.txt_eos_index,
                    pad_index=self.txt_pad_index,
                    bos_index=self.txt_bos_index,
                    decoder=self.decoder,
                )
        else:
            stacked_txt_output = stacked_attention_scores = None

        return decoded_gloss_sequences, stacked_txt_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return (
            "%s(\n"
            "\tencoder=%s,\n"
            "\tdecoder=%s,\n"
            "\tsgn_embed=%s,\n"
            "\ttxt_embed=%s)"
            % (
                self.__class__.__name__,
                self.encoder,
                self.decoder,
                self.sgn_embed,
                self.txt_embed,
            )
        )


def build_model(
    cfg: dict,
    sgn_dim: int,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    do_recognition: bool = True,
    do_translation: bool = True,
) -> SignModel:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param sgn_dim: feature dimension of the sign frame representation, i.e. 2560 for EfficientNet-7.
    :param gls_vocab: sign gloss vocabulary
    :param txt_vocab: spoken language word vocabulary
    :return: built and initialized model
    :param do_recognition: flag to build the model with recognition output.
    :param do_translation: flag to build the model with translation decoder.
    """

    txt_padding_idx = txt_vocab.stoi[PAD_TOKEN]

    sgn_embed: SpatialEmbeddings = SpatialEmbeddings(
        **cfg["encoder"]["embeddings"],
        num_heads=cfg["encoder"]["num_heads"],
        input_size=sgn_dim,
    )

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.0)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert (
            cfg["encoder"]["embeddings"]["embedding_dim"]
            == cfg["encoder"]["hidden_size"]
        ), "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )
    else:
        encoder = RecurrentEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )

    if do_recognition:
        gloss_output_layer = nn.Linear(encoder.output_size, len(gls_vocab))
        if cfg["encoder"].get("freeze", False):
            freeze_params(gloss_output_layer)
    else:
        gloss_output_layer = None

    # build decoder and word embeddings
    if do_translation:
        txt_embed: Union[Embeddings, None] = Embeddings(
            **cfg["decoder"]["embeddings"],
            num_heads=cfg["decoder"]["num_heads"],
            vocab_size=len(txt_vocab),
            padding_idx=txt_padding_idx,
        )
        dec_dropout = cfg["decoder"].get("dropout", 0.0)
        dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
        if cfg["decoder"].get("type", "recurrent") == "transformer":
            decoder = TransformerDecoder(
                **cfg["decoder"],
                encoder=encoder,
                vocab_size=len(txt_vocab),
                emb_size=txt_embed.embedding_dim,
                emb_dropout=dec_emb_dropout,
            )
        else:
            decoder = RecurrentDecoder(
                **cfg["decoder"],
                encoder=encoder,
                vocab_size=len(txt_vocab),
                emb_size=txt_embed.embedding_dim,
                emb_dropout=dec_emb_dropout,
            )
    else:
        txt_embed = None
        decoder = None

    model: SignModel = SignModel(
        encoder=encoder,
        gloss_output_layer=gloss_output_layer,
        decoder=decoder,
        sgn_embed=sgn_embed,
        txt_embed=txt_embed,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        do_recognition=do_recognition,
        do_translation=do_translation,
    )

    if do_translation:
        # tie softmax layer with txt embeddings
        if cfg.get("tied_softmax", False):
            # noinspection PyUnresolvedReferences
            if txt_embed.lut.weight.shape == model.decoder.output_layer.weight.shape:
                # (also) share txt embeddings and softmax layer:
                # noinspection PyUnresolvedReferences
                model.decoder.output_layer.weight = txt_embed.lut.weight
            else:
                raise ValueError(
                    "For tied_softmax, the decoder embedding_dim and decoder "
                    "hidden_size must be the same."
                    "The decoder must be a Transformer."
                )

    # custom initialization of model parameters
    initialize_model(model, cfg, txt_padding_idx)

    return model

def distillation_loss(student_logits, teacher_logits, T, alpha):
    """
    Calculate the distillation loss between the student's logits and the teacher's logits.
    """
    soft_student_logits = F.log_softmax(student_logits / T, dim=-1)
    soft_teacher_logits = F.softmax(teacher_logits / T, dim=-1)
    distill_loss = F.kl_div(soft_student_logits, soft_teacher_logits, reduction="batchmean") * (T ** 2)
    return distill_loss
    
def load_teacher_model(ckpt_path, cfg_file, gls_vocab, txt_vocab):
    cfg = load_config(cfg_file)
    model_checkpoint = load_checkpoint(ckpt_path, torch.cuda.is_available())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        do_recognition=True,
        do_translation=True,
    ).to(device)

    teacher_model.load_state_dict(model_checkpoint["model_state"])
    
    return teacher_model

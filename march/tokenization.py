import march  # Redirect cache

from os.path import join

from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from datasets import load_dataset

from transformers import PreTrainedTokenizerFast

from march import CONFIG_DIR


__all__ = ["EOS_TOKEN", "EXTRA_ID_TOKENS", "train_tokenizer", "load_tokenizer"]

EOS_TOKEN = "</s>"
EXTRA_ID_TOKENS = [f"<extra_id_{i}>" for i in reversed(range(100))]
TOKENIZER_FILE = join(CONFIG_DIR, "tokenizer.json")
MAX_LENGTH = 1024
VOCAB_SIZE = 30000


def train_tokenizer() -> None:
    tokenizer = Tokenizer(BPE(byte_fallback=True))
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=3,
        show_progress=True,
        special_tokens=[EOS_TOKEN] + EXTRA_ID_TOKENS + list("0123456789"),
    )
    tokenizer.pre_tokenizer = Whitespace()

    train_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")["train"]["text"]

    tokenizer.train_from_iterator(train_dataset, trainer=trainer, length=len(train_dataset))

    tokenizer.post_processor = TemplateProcessing(
        single=f"$A {EOS_TOKEN}",
        pair=f"$A {EOS_TOKEN} $B {EOS_TOKEN}",
        special_tokens=[
            (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
        ],
    )
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id(EOS_TOKEN), pad_token=EOS_TOKEN)
    tokenizer.enable_truncation(MAX_LENGTH)
    tokenizer.save(TOKENIZER_FILE)


def load_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(TOKENIZER_FILE))
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)

    return tokenizer
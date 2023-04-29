import march  # Redirect cache

from os.path import join

from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from datasets import load_dataset, Dataset

from transformers import PreTrainedTokenizerFast

from march import CONFIG_DIR


EOS_TOKEN = "</s>"
EXTRA_ID_TOKENS = [f"<extra_id_{i}>" for i in reversed(range(100))]
MAX_LENGTH = 1024

WIKITEXT_VOCAB_SIZE = 30000
WIKITEXT_TOKENIZER_FILE = join(CONFIG_DIR, "tokenizer_wikitext_103.json")

C4_VOCAB_SIZE = 32000
C4_TOKENIZER_FILE = join(CONFIG_DIR, "tokenizer_c4.json")


def train_tokenizer(train_dataset: Dataset, tokenizer_path: str, vocab_size: int) -> None:
    tokenizer = Tokenizer(BPE(byte_fallback=True))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=3,
        show_progress=True,
        special_tokens=[EOS_TOKEN] + EXTRA_ID_TOKENS + list("0123456789"),
    )
    tokenizer.pre_tokenizer = Whitespace()

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
    tokenizer.save(tokenizer_path)


def train_wikitext_tokenizer() -> None:
    wikitext_train_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")["train"]["text"]
    train_tokenizer(wikitext_train_dataset, WIKITEXT_TOKENIZER_FILE, WIKITEXT_VOCAB_SIZE)


def load_wikitext_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(WIKITEXT_TOKENIZER_FILE))
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)

    return tokenizer


def train_c4_tokenizer() -> None:
    c4_train_dataset = load_dataset("c4", "en")["train"]
    train_tokenizer(c4_train_dataset, C4_TOKENIZER_FILE, C4_VOCAB_SIZE)


def load_c4_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(C4_TOKENIZER_FILE))
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)

    return tokenizer

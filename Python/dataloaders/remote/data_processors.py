import transformers
from dataloaders.remote.language_modelers import LanguageModelerScheme

class DataProcessor:
    def __init__(self, model_name=str, max_sequence_size=int, chunking=False):
        if chunking and max_sequence_size % 128 != 0:
            raise EnvironmentError("max_sequence_size must be a multiple of 128 in chunking mode.")

        self.chunking = chunking
        self.max_sequence_size = max_sequence_size
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.lm_scheme = LanguageModelerScheme("t5")
        self.attention_masks = False
        self.token_types = False
        self.labels = False
        self.decoder_inputs = False

    def set_fields(self, attention_masks=False, token_types=False, labels=False, decoder_inputs=False):
        self.attention_masks = attention_masks
        self.token_types = token_types
        self.labels = labels
        self.decoder_inputs = decoder_inputs

    def set_scheme(self, lm_scheme=str):


    def process_row(self, row):
        pass
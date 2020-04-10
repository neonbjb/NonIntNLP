import torch

class LanguageModelerScheme:
    def __init__(self, scheme_name):
        self.load_scheme(scheme_name)
        pass

    def load_scheme(self, scheme_name=str):
        # Validate and save it.
        pass

    def corrupt_and_label(self, tokenized_string=torch.tensor):
        # return (corrupted_string, label)
        pass

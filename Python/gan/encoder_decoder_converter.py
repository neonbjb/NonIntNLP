import transformers
import torch


# Transformers encoder and decoder layers do not share the same embedding space. Fortunately, you can (generally?) train
# a linear function that maps from one space to another. This class uses that function to allow you to convert from
# the encoder space to the decoder space and vice versa.
class EncoderDecoderConverter(torch.nn.Module):
    def __init__(self, pretrained_layer_path, forward_enc_to_dec=True, device="cuda"):
        super(EncoderDecoderConverter, self).__init__()
        self.conversion_layer = torch.load(pretrained_layer_path).to(device)
        self.forward_enc_to_dec = forward_enc_to_dec

    def forward(self, x):
        if self.forward_enc_to_dec:
            return self.encoder_to_decoder(x)
        else:
            return self.decoder_to_encoder(x)

    def encoder_to_decoder(self, tensor):
        return self.conversion_layer(tensor)

    def decoder_to_encoder(self, tensor):
        if self.conversion_layer.bias is not None:
            tensor -= self.conversion_layer.bias
        return tensor.matmul(self.conversion_layer.weight.inverse().t())


if __name__ == "__main__":
    conv = EncoderDecoderConverter("C:\\Users\\jbetk\\Documents\\data\\ml\\saved_models\\ganformer\\converters\\encoder_decoder.pt", device="cpu")
    tokenizer = transformers.XLNetTokenizer.from_pretrained("xlnet-base-cased")
    xlnet = transformers.XLNetLMHeadModel.from_pretrained("xlnet-base-cased")
    enc = xlnet.get_input_embeddings()
    dec = xlnet.get_output_embeddings()

    string = "Let's see how well this gets converted.<pad></s>"
    toked = torch.tensor(tokenizer.encode(string), dtype=torch.long).unsqueeze(0)

    f = dec(conv.encoder_to_decoder(enc(toked)))
    print(tokenizer.decode(f.softmax(-1).argmax(-1)[0]))

    b = dec(conv.encoder_to_decoder(conv.decoder_to_encoder(conv.encoder_to_decoder(enc(toked)))))
    print(tokenizer.decode(b.softmax(-1).argmax(-1)[0]))



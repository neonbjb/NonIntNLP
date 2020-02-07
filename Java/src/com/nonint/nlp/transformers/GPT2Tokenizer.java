package com.nonint.nlp.transformers;

import java.io.IOException;
import java.text.ParseException;
import java.util.Map;

public class GPT2Tokenizer extends Tokenizer {
    private GPT2Tokenizer(String modelName) {
        super(modelName, "right");
    }

    public static GPT2Tokenizer fromPretrained(String modelName) throws IOException, ParseException {
        GPT2Tokenizer result = new GPT2Tokenizer(modelName);
        result.loadRequiredFiles();
        // Left off at line 435

        return result;
    }

    public static void main(String[] args) throws Exception {
        FileUtil.initSsl();
        fromPretrained("gpt2");
    }

    @Override
    protected Map<String, String> vocabFilesNames() {
        return Map.of("vocab_file", "vocab.json",
                "merges_file", "merges.txt");
    }

    @Override
    protected Map<String, Map<String, String>> pretrainedVocabFilesMap() {
        return Map.of(
                "vocab_file",
                Map.of(
                    "gpt2", "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
                    "gpt2-medium", "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-vocab.json",
                    "gpt2-large", "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-vocab.json",
                    "gpt2-xl","https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-vocab.json",
                    "distilgpt2", "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-vocab.json"
                ),
                "merges_file",
                Map.of(
                    "gpt2", "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
                    "gpt2-medium", "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-merges.txt",
                    "gpt2-large", "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-merges.txt",
                    "gpt2-xl", "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-merges.txt",
                    "distilgpt2", "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-merges.txt"
                )
        );
    }

    @Override
    protected Map<String, Integer> pretrainedPositionalEmbeddingsSizes() {
        return Map.of(
            "gpt2", 1024,
            "gpt2-medium", 1024,
            "gpt2-large", 1024,
            "gpt2-xl", 1024,
            "distilgpt2", 1024
        );
    }
}

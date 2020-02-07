package com.nonint.nlp.transformers;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import javax.net.ssl.HttpsURLConnection;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.*;
import java.util.logging.Logger;

public abstract class Tokenizer {
    protected abstract Map<String, String> vocabFilesNames();
    protected abstract Map<String, Map<String, String>> pretrainedVocabFilesMap();
    protected abstract Map<String, Integer> pretrainedPositionalEmbeddingsSizes();

    String cacheDir = "tokenizer-cache";
    Map<String, File> resolvedVocabFiles = new HashMap<>();

    JSONObject config;
    String modelName;
    String paddingSide;

    public Optional<String> bosToken = Optional.empty();
    public Optional<String> eosToken = Optional.empty();
    public Optional<String> unkToken = Optional.empty();
    public Optional<String> sepToken = Optional.empty();
    public Optional<String> padToken = Optional.empty();
    public Optional<String> clsToken = Optional.empty();
    public Optional<String> maskToken = Optional.empty();
    public List<String> additionalSpecialTokens = new ArrayList<>();

    int bosTokenId() {
        return convertTokensToIds(bosToken.get());
    }

    int eosTokenId() {
        return convertTokensToIds(eosToken.get());
    }

    int unkTokenId() {
        return convertTokensToIds(unkToken.get());
    }

    int sepTokenId() {
        return convertTokensToIds(sepToken.get());
    }

    int padTokenId() {
        return convertTokensToIds(padToken.get());
    }

    int clsTokenId() {
        return convertTokensToIds(clsToken.get());
    }

    int maskTokenId() {
        return convertTokensToIds(maskToken.get());
    }

    List<Integer> additionalSpecialTokenIds() {
        return convertTokensToIds(additionalSpecialTokens);
    }

    protected Tokenizer(String modelName, String paddingSide) {
        this.modelName = modelName;
        this.paddingSide = paddingSide;
    }

    protected void loadRequiredFiles() throws IOException, ParseException {
        final String[] FILES = new String[] { SPECIAL_TOKENS_MAP_FILE, ADDED_TOKENS_FILE, TOKENIZER_CONFIG_FILE };
        for(String file : FILES) {
            Path cachedPath = Paths.get(cacheDir + "/" + file);
            if(Files.notExists(cachedPath)) {
                URL url = FileUtil.hfBucketUrl(modelName, file);
                Logger.getGlobal().info("Downloading " + url.toString());
                InputStream in = ((HttpsURLConnection)url.openConnection()).getInputStream();
                Files.copy(in, Paths.get(cacheDir + "/" + file), StandardCopyOption.REPLACE_EXISTING);
            }
            resolvedVocabFiles.put(file, cachedPath.toFile());
        }

        config = (JSONObject)new JSONParser().parse(new FileReader(resolvedVocabFiles.get(TOKENIZER_CONFIG_FILE)));
        Logger.getGlobal().info("Done loading files.");
    }

    protected int convertTokensToIds(String token) {
        return 0;
    }

    protected List<Integer> convertTokensToIds(List<String> tokens) {
        List<Integer> result = new ArrayList<>();

        return result;
    }
}

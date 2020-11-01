# Using marian with factors

Following this README should allow the user to train a model with source- and/or target-side factors. To train with factors, the data must be formatted in a certain way. A special vocabulary file format is also required. See details below.

### Requirements:

[Marian CEF marian-dev](https://github.com/marian-cef/marian-dev)


## Define factors

Factors should be organized in "groups," where each group represents a different feature. For example, there could be a group denoting capitalization and another denoting subword divisions.

Factors within a single group should start with the same string.

For example, for a capitalization factor group, the individual factors could be:

`c0`: all lowercase

`c1`: first character capitalized, rest lowercase

`c2`: all uppercase

If there were a second factor group for subword divisions, the individual factors could be:

`s0`: end of word, whitespace should follow

`s1`: join token with next subword

There is no limit on the number of factor groups barring some practical limitations having to do with how the vocabulary is stored by `marian`. If the limit is exceeded `marian` will throw this error.

Factor group zero is always the actual words in the text, referred to as *lemmas*.

## Data preparation

Factors are appended to the *lemmas* with a pipe `|`. The pipe also separates factors of multiple groups.

Example sentence:

```
Trump tested positive for COVID-19.
```

Preprocessed sentence:
```
trump test@@ ed positive for c@@ o@@ v@@ i@@ d - 19 .
```

Apply factors:
```
trump|c1|s0 test|c0|s1 ed|c0|s0 positive|c0|s0 for|c0|s0 c|c2|s1 o|c2|s1 v|c2|s1 i|c2|s1 d|c2|s0 -|c0|s0 19|c0|s0 .|c0|s0
```


## Create the factored vocabulary

Factored vocabularies should have the extension `.fsv`. How to structure the vocabulary file is described below. If using factors only on the source or target side, the vocabulary of the other side can be a normal `json`, `yaml`, etc. 

The `.fsv` vocabulary should have three sections:

1. **Factors**

    The factor groups are defined with an underscore prepended. The colon indicates which factor group each factor inherits from. `_has_c` is used in the definition of the words in the vocabulary (see #2 below) to indicate that that word has that factor group. The `_lemma` factor is used for the words/tokens themselves; this must be present. 

    ```
    _lemma

    _c
    c0 : _c
    c1 : _c
    c2 : _c
    _has_c

    _s
    s0 : _s
    s1 : _s
    _has_s
    ```

2. **Lemmas**

    These are the vocabulary entries themselves. They have the format of `LEMMA : _lemma [_has_c] [_has_s]`. The `_has_X` should only apply to lemmas that can have an `X` factor anywhere in the data (which will likely be all of the tokens except `</s>` and `<unk>`).

    Examples:
    ```
    </s> : _lemma
    <unk> : _lemma
    , : _lemma _has_c _has_s
    . : _lemma _has_c _has_s
    the : _lemma _has_c _has_s
    for: _lemma _has_c _has_s
    ```


#### Other suggestions

Certain characters are used by the `.fsv` vocabulary that will have to be escaped/replaced in the data: `#:_\|`

The tokens in the factor vocabularies (`c0`, `c1`, `s0`, etc.) cannot be present in any of the *lemmas*.

### Full `.fsv` file

Putting everything together, the final `.fsv` file should look like this. It can have comments.

 ```
 # factors

_lemma

_c
c0 : _c
c1 : _c
c2 : _c
_has_c

_s
s0 : _s
s1 : _s
_has_s

 # lemmas

</s> : _lemma
<unk> : _lemma
, : _lemma _has_c _has_s
. : _lemma _has_c _has_s
the : _lemma _has_c _has_s
for: _lemma _has_c _has_s
 ```

## Training options

There are two choices for how factor embeddings are combined with *lemma* embeddings: summation and concatenation.

```
--factors-combine TEXT=sum                      How to combine the factors and lemma embeddings.
                                                Options available: sum, concat
```

The dimension of the factor embeddings must be specified if using combine option `concat`. If using `sum`, the factor embedding dimension matches that of the lemmas.

```
--factors-dim-emb INT                           Embedding dimension of the factors. Only used if 
                                                concat is selected as factors combining form
```

### Prediction

If using factors on the target side, there are multiple options for how factor predictions are generated related to the form of conditioning / dependencies of factors and lemmas:

```
--factor-predictor TEXT=soft-transformer-layer  Method to use when predicting target factors. 
                                                Options: soft-transformer-layer, hard-transformer-layer,
                                                lemma-dependent-bias, re-embedding
--lemma-dim-emb INT=0                           Re-embedding dimension of lemma in factors
```

* `soft-transformer-layer`: Uses an additional transformer layer to predict the factors using the previously predicted lemma
* `hard-transformer-layer`: Like `soft-transformer-layer` but with hard-max
* `lemma-dependent-bias`: Adds a learned bias term based on the predicted lemma to the logits of the factors. There is no additional transformer layer introduced with this option
* `re-embedding`: After predicting a lemma, re-embed the lemma and add this new vector before predicting the factors
* `lemma-dim-emb`: Controls the dimension of the re-embedded lemma when using the option `re-embedding`


### Weight tying

If using factors only on the source or target side but using a joint vocabulary, there are two options to tie source and target embedding weights:

1. Use combine option `concat`
2. Create "dummy" factors for the side initially without factors. This entails creating a factored vocabulary where the same number of factors are present as are on the side with meaningful factors. In the previous example, if we have the capitalization and subword factors on the source side, the target side would have five different dummy factors (they can all be in the same group). In the *lemma* section of the `.fsv` file we would just not put `_has_X` for any lemma.

    ```
    # factors

    _lemma

    _d
    d0 : _d
    d1 : _d
    d2 : _d
    d3 : _d
    d4 : _d
    _has_d

    # lemmas

    </s> : _lemma
    <unk> : _lemma
    , : _lemma
    . : _lemma
    le : _lemma
    pour: _lemma
    ```
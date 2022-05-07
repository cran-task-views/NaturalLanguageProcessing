---
name: NaturalLanguageProcessing
topic: Natural Language Processing
maintainer: Fridolin Wild
email: wild@open.ac.uk
version: 2022-05-06
source: https://github.com/cran-task-views/NaturalLanguageProcessing/
---


Natural language processing has come a long way since its foundations
were laid in the 1940s and 50s (for an introduction see, e.g., Jurafsky
and Martin (2008, 2009, 2022 draft third edition): Speech and Language Processing, Pearson Prentice
Hall). This CRAN task view collects relevant R packages that support
computational linguists in conducting analysis of speech and language on
a variety of levels - setting focus on words, syntax, semantics, and
pragmatics.

In recent years, we have elaborated a framework to be used in packages
dealing with the processing of written material: the package
`r pkg("tm", priority = "core")`. Extension packages in this
area are highly recommended to interface with tm's basic routines and
useRs are cordially invited to join in the discussion on further
developments of this framework package. 

A basic introduction with comprehensive examples is provided in the
book by Fridolin Wild (2016): Learning Analytics in R, Springer. 

#### Frameworks:

-   `r pkg("tm")` provides a comprehensive text mining
    framework for R. The [Journal of Statistical
    Software](http://www.jstatsoft.org/) article [Text Mining
    Infrastructure in R](http://www.jstatsoft.org/v25/i05/) gives a
    detailed overview and presents techniques for count-based analysis
    methods, text clustering, text classification and string kernels.
-   `r pkg("tm.plugin.dc")` allows for distributing corpora
    across storage devices (local files or Hadoop Distributed File
    System).
-   `r pkg("tm.plugin.mail")` helps with importing mail
    messages from archive files such as used in Thunderbird (mbox, eml).
-   `r pkg("tm.plugin.alceste")` allows importing text
    corpora written in a file in the Alceste format.
-   `r pkg("tm.plugin.webmining")` allow importing news
    feeds in XML (RSS, ATOM) and JSON formats. Currently, the following
    feeds are implemented: Google Blog Search, Google Finance, Google
    News, NYTimes Article Search, Reuters News Feed, Yahoo Finance, and
    Yahoo Inplay.
-   `r pkg("RcmdrPlugin.temis")` is an Rcommander plug-in
    providing an integrated solution to perform a series of text mining
    tasks such as importing and cleaning a corpus, and analyses like
    terms and documents counts, vocabulary tables, terms co-occurrences
    and documents similarity measures, time series analysis,
    correspondence analysis and hierarchical clustering.
-   `r pkg("openNLP")` provides an R interface to
    [OpenNLP](http://opennlp.sourceforge.net/), a collection of natural
    language processing tools including a sentence detector, tokenizer,
    pos-tagger, shallow and full syntactic parser, and named-entity
    detector, using the Maxent Java package for training and using
    maximum entropy models.
-   Trained models for English and Spanish to be used with
    `r pkg("openNLP")` are available from
    <http://datacube.wu.ac.at/> as packages openNLPmodels.en and
    openNLPmodels.es, respectively.
-   `r pkg("RWeka")` is a interface to
    [Weka](http://www.cs.waikato.ac.nz/ml/weka/) which is a collection
    of machine learning algorithms for data mining tasks written in
    Java. Especially useful in the context of natural language
    processing is its functionality for tokenization and stemming.
-   `r pkg("tidytext")` provides means for text mining for
    word processing and sentiment analysis using dplyr, ggplot2, and
    other tidy tools.
-   `r pkg("udpipe")` provides language-independant
    tokenization, part of speech tagging, lemmatization, dependency
    parsing, and training of treebank-based annotation models.

#### Words (lexical DBs, keyword extraction, string manipulation, stemming)

-   R's base package already provides a rich set of character
    manipulation routines. See
    `help.search(keyword = "character", package = "base")` for more
    information on these capabilities.
-   `r pkg("wordnet")` provides an R interface to
    [WordNet](http://wordnet.princeton.edu/), a large lexical database
    of English.
-   `r pkg("RKEA")` provides an R interface to
    [KEA](http://www.nzdl.org/Kea/) (Version 5.0). KEA (for Keyphrase
    Extraction Algorithm) allows for extracting keyphrases from text
    documents. It can be either used for free indexing or for indexing
    with a controlled vocabulary.
-   `r pkg("gsubfn")` can be used for certain parsing tasks
    such as extracting words from strings by content rather than by
    delimiters. `demo("gsubfn-gries")` shows an example of this in a
    natural language processing context.
-   `r pkg("textreuse")` provides a set of tools for
    measuring similarity among documents and helps with detecting
    passages which have been reused. The package implements shingled
    n-gram, skip n-gram, and other tokenizers; similarity/dissimilarity
    functions; pairwise comparisons; minhash and locality sensitive
    hashing algorithms; and a version of the Smith-Waterman local
    alignment algorithm suitable for natural language.
-   `r pkg("boilerpipeR")` helps with the extraction and
    sanitizing of text content from HTML files: removal of ads,
    sidebars, and headers using the boilerpipe Java library.
-   `r pkg("tau")` contains basic string manipulation and
    analysis routines needed in text processing such as dealing with
    character encoding, language, pattern counting, and tokenization.
-   `r pkg("SnowballC")` provides exactly the same API as
    Rstem, but uses a slightly different design of the C libstemmer
    library from the Snowball project. It also supports two more
    languages.
-   `r pkg("stringi")` provides R language wrappers to the
    International Components for Unicode (ICU) library and allows for:
    conversion of text encodings, string searching and collation in any
    locale, Unicode normalization of text, handling texts with mixed
    reading direction (e.g., left to right and right to left), and text
    boundary analysis (for tokenizing on different aggregation levels or
    to identify suitable line wrapping locations).
-   `r pkg("stringdist")` implements an approximate string
    matching version of R's native 'match' function. It can calculate
    various string distances based on edits (Damerau-Levenshtein,
    Hamming, Levenshtein, optimal string alignment), qgrams (q-gram,
    cosine, jaccard distance) or heuristic metrics (Jaro, Jaro-Winkler).
    An implementation of soundex is provided as well. Distances can be
    computed between character vectors while taking proper care of
    encoding or between integer vectors representing generic sequences.
-   `r ohat("Rstem")` (available from Omegahat) is an
    alternative interface to a C version of Porter's word stemming
    algorithm.
-   `r pkg("koRpus")` is a diverse collection of functions
    for automatic language detection, hyphenation, several indices of
    lexical diversity (e.g., type token ratio, HD-D/vocd-D, MTLD) and
    readability (e.g., Flesch, SMOG, LIX, Dale-Chall). See the [web
    page](http://reaktanz.de/?c=hacking&s=koRpus) for more information.
-   `r pkg("ore")` provides an alternative to R's built-in
    functionality for handling regular expressions, based on the Onigmo
    Regular Expression Library. Offers first-class compiled regex
    objects, partial matching and function-based substitutions, amongst
    other features. A benchmark comparing results for ore functions with
    stringi and the R base implementation is available
    `r github("jonclayden/regex-performance")`.
-   `r pkg("languageR")` provides data sets and functions
    exemplifying statistical methods, and some facilitatory utility
    functions used in the book by R. H. Baayen: "Analyzing Linguistic
    Data: a Practical Introduction to Statistics Using R", Cambridge
    University Press, 2008.
-   `r pkg("zipfR")` offers some statistical models for word
    frequency distributions. The utilities include functions for
    loading, manipulating and visualizing word frequency data and
    vocabulary growth curves. The package also implements several
    statistical models for the distribution of word frequencies in a
    population. (The name of this library derives from the most famous
    word frequency distribution, Zipf's law.)
-   `r pkg("wordcloud")` provides a visualisation similar to
    the famous wordle ones: it horizontally and vertically distributes
    features in a pleasing visualisation with the font size scaled by
    frequency.
-   `r pkg("hunspell")` is a stemmer and spell-checker
    library designed for languages with rich morphology and complex word
    compounding or character encoding. The package can check and analyze
    individual words as well as search for incorrect words within a
    text, latex or (R package) manual document.
-   `r pkg("phonics")` provides a collection of phonetic
    algorithms including Soundex, Metaphone, NYSIIS, Caverphone, and
    others.
-   `r pkg("tesseract")` is an OCR engine with unicode
    (UTF-8) support that can recognize over 100 languages out of the
    box.
-   `r pkg("mscsweblm4r")` provides an interface to the
    Microsoft Cognitive Services Web Language Model API and can be used
    to calculate the probability for a sequence of words to appear
    together, the conditional probability that a specific word will
    follow an existing sequence of words, get the list of words
    (completions) most likely to follow a given sequence of words, and
    insert spaces into a string of words adjoined together without any
    spaces (hashtags, URLs, etc.).
-   `r pkg("mscstexta4r")` provides an interface to the
    Microsoft Cognitive Services Text Analytics API and can be used to
    perform sentiment analysis, topic detection, language detection, and
    key phrase extraction.
-   `r pkg("sentencepiece")` is an unsupervised tokeniser producing Byte Pair Encoding
    (BPE), Unigram, Char, or Word models.
-   `r pkg("tokenizers")` helps split text into tokens,
    supporting shingled n-grams, skip n-grams, words, word stems,
    sentences, paragraphs, characters, lines, and regular expressions.
-   `r pkg("tokenizers.bpe")` helps split text into syllable
    tokens, implemented using Byte Pair Encoding and the YouTokenToMe
    library.
-   `r pkg("crfsuite")` uses Conditional Random Fields for
    labelling sequential data.

#### Semantics:

-   `r pkg("lsa")` provides routines for performing a latent
    semantic analysis with R. The basic idea of latent semantic analysis
    (LSA) is, that text do have a higher order (=latent semantic)
    structure which, however, is obscured by word usage (e.g. through
    the use of synonyms or polysemy). By using conceptual indices that
    are derived statistically via a truncated singular value
    decomposition (a two-mode factor analysis) over a given
    document-term matrix, this variability problem can be overcome. The
    article [Representing and Analysing Meaning with LSA](https://doi.org/10.1007/978-3-319-28791-1_4) by Wild (2016)
    gives a detailed overview and comprehensive examples.
-   `r pkg("topicmodels")` provides an interface to the C
    code for Latent Dirichlet Allocation (LDA) models and Correlated
    Topics Models (CTM) by David M. Blei and co-authors and the C++ code
    for fitting LDA models using Gibbs sampling by Xuan-Hieu Phan and
    co-authors.
-   `r pkg("BTM")` helps identify topics in texts from
    term-term cooccurrences (hence 'biterm' topic model, BTM).
-   `r pkg("topicdoc")` provides topic-specific diagnostics
    for LDA and CTM topic models to assist in evaluating topic quality.
-   `r pkg("lda")` implements Latent Dirichlet Allocation
    and related models similar to LSA and topicmodels.
-   `r pkg("stm")` (Structural Topic Model) implements a
    topic model derivate that can include document-level meta-data. The
    package also includes tools for model selection, visualization, and
    estimation of topic-covariate regressions.
-   `r pkg("kernlab")` allows to create and compute with
    string kernels, like full string, spectrum, or bounded range string
    kernels. It can directly use the document format used by
    `r pkg("tm")` as input.
-   `r github("bnosac/golgotha")` (not yet on CARN) provides
    a wrapper to Bidirectional Encoder Representations from Transformers
    (BERT) for language modelling and textual entailment in particular.
-   `r pkg("ruimtehol")` provides a neural network machine
    learning approach to vector space semantics, implementing an
    interface to StarSpace, providing means for classification,
    proximity measurement, and model management (training, predicting,
    several interfaces for textual entailment of varying granularity).
-   `r pkg("skmeans")` helps with clustering providing
    several algorithms for spherical k-means partitioning.
-   `r pkg("movMF")` provides another clustering alternative
    (approximations are fitted with von Mises-Fisher distributions of
    the unit length vectors).
-   `r pkg("textir")` is a suite of tools for text and
    sentiment mining.
-   `r pkg("textcat")` provides support for n-gram based
    text categorization.
-   `r pkg("textrank")` is an extension of the PageRank and
    allows to summarize text by calculating how sentences are related to
    one another.
-   `r pkg("corpora")` offers utility functions for the
    statistical analysis of corpus frequency data.
-   `r pkg("text2vec")` provides tools for text
    vectorization, topic modeling (LDA, LSA), word embeddings (GloVe),
    and similarities.
-   `r pkg("word2vec")` allows to learn vector representations of 
    words by continuous bag of words and skip-gram implementations of the
    "word2vec" algorithm. The techniques are detailed in the paper
    [Distributed Representations of Words and Phrases and their 
    Compositionality](https://arxiv.org/abs/1310.4546) by Mikolov et al. (2013).

#### Pragmatics:

-   `r pkg("qdap")` helps with quantitative discourse
    analysis of transcripts.
-   `r pkg("quanteda")` supports quantitative analysis of
    textual data.

#### Corpora:

-   `r pkg("corporaexplorer")` facilitates visual
    information retrieval over document collections, supporting
    filtering and corpus-level as well as document-level visualisation
    using an interactive web apps built using Shiny.
-   `r pkg("textplot")` provides various methods for
    corpus-, document-, and sentence-level visualisation.
-   `r pkg("tm.plugin.factiva")`,
    `r pkg("tm.plugin.lexisnexis")`,
    `r pkg("tm.plugin.europresse")` allow importing press
    and Web corpora from (respectively) Dow Jones Factiva, LexisNexis,
    and Europresse.



### Links
-   [Learning Analytics in R: with tutorial style examples for tm, lsa, mpia, network)](https://link.springer.com/book/10.1007/978-3-319-28791-1)
-   [A Gentle Introduction to Statistics for (Computational) Linguists (SIGIL)](http://www.stefan-evert.de/SIGIL/)
-   [Stefan Th. Gries (2017): Quantitative Corpus Linguistics with R, 2nd ed., Routledge.](https://www.routledge.com/9781138816282)
-   [Dan Jurafsky and James H. Martin (2022): Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/)
-   [Corpora and NLP model packages at http://datacube.wu.ac.at/](http://datacube.wu.ac.at/)

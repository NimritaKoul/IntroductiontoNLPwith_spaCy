# Title    : Introduction to Natural Language Processing with spaCy
### Author   : Dr. Nimrita Koul, Associate Professor, Machine Learning, Bangalore, India.
### Github   :  
### LinkedIn:

#### References:
1. https://spacy.io/usage/spacy-101
2. https://course.spacy.io/en/
3. https://github.com/explosion/spacy-course/tree/master/exercises/en
3. https://www.machinelearningplus.com/spacy-tutorial-nlp/
4. https://blog.dominodatalab.com/natural-language-in-python-using-spacy


#### Pre-requisites for this attending this session: 

1. Intermediate knowledge of Python 3 programming and Jupyter Notebook application.
2. Knowledge of libraries - pandas, numpy their installation using pip or conda

#### After attending this session, the participants will know the answers to these questions:
1. What is Natural Language Processing?
2. What is spaCy - Architecture, Features and Functionality?
3. How to do NLP tasks in spaCy?

### 1. What is Natural Language Processing?

- A Natural Language is a language like English, Hindi, German etc. that is used by "Humans" to communicate with one another. 


- Natural language processing (NLP) is a branch of artificial intelligence that aims at building computer systems that can understand natural language (in text or speech) and respond to in natural language. 

- To "understand" means to recognize the meaning, the intent and the sentiment of the natural language text or speech. 

- NLP makes use of statistical models, machine learning, deep learning to discover insights from your natural language data. 
  It helps you answer questions like:

        1. What is the central idea of the text?
        2. Who is doing what to whom?
        3. Is the customer happy or upset about the product or service?
        4. How similar are there two pieces of text?

- It has applications in automated voice response systems, smart assistants like Alexa, chatbots, machine translation etc like Google Translate etc. 

####  Understanding natural languages involves tasks like- 

        a. Identifying nouns, verbs and more,  i.e. parts of speech. 
        b. Identifying relationships among nouns and verbs. (Who is doing what to whom?)
        c. Disambiguating meaning of a word in a context. 
        d. Identifying real world Entities. (Names of people, countries, companies)
        e. Sentiment Analysis. (Is the customer happy or upset about the product?) 

 
### 2. [What is spaCy?](https://spacy.io/usage/spacy-101)

    - spaCy is a free, open-source, high speed library for advanced Natural Language Processing (NLP) in Python.  
    - Designed for building productions level NLP systems for information extraction, text processing and understanding.
    - Can handle large volumes of text.
    - Supports 66+ languages and provides 76 trained NLP Pipelines/Models for 23 languages.
    - Provides pretrained transformers like BERT.
    - Extensible with your own components and attributes, supports custom models in frameworks like PyTorch, TensorFlow.
    
    

### 2a. Important Functionality of spaCy

|NAME	| DESCRIPTION| 
|:-----|:------------|
|Tokenization	| Segmenting text into words, punctuations marks etc.|
|Part-of-speech (POS) Tagging |	Assigning word types to tokens, like verb or noun.|
|Dependency Parsing|	Assigning syntactic dependency labels, describing the relations between individual tokens, like subject or object.|
|Lemmatization|	Assigning the base forms of words. For example, the lemma of “rats” is “rat”, of "reading" is "read"|
|Sentence Boundary Detection (SBD)|	Finding and segmenting individual sentences.|
|Named Entity Recognition (NER)|	Identifying and labelling named “real-world” objects, like persons, companies, locations.|
|Entity Linking (EL)|	Disambiguating textual entities to unique identifiers in a knowledge base.|
|Similarity|	Comparing words, text spans and documents and how similar they are to each other.|
|Text Classification|	Assigning categories or labels to a whole document, or parts of a document.|
|Rule-based Matching|	Finding tokens that match a specific lexical pattern or linguistic annotaions.|
|Training	|Updating and improving a statistical model’s predictions.|
|Serialization|	Saving objects to files or byte strings.|


### 2b. [Trained Statistical models in spaCy](https://spacy.io/usage/models)

- spaCy provides 76 trained pipelines containing statistical models to predict linguistic annotations of your text in about 23 languages. These linguistic annotations include information like POS tags, dependency, entity names for your text. 

## 3. Installing spaCy

#### My preferred way of using Python is its [Anaconda Distribution](https://www.anaconda.com/products/distribution). I prefer to code using Jupyter Notebook. That is what is used for this tutorial.

#### 3a. Install spacy using Pip

<div class = "alert alert-block alert-info">
     <b>It is recommended to create a separate virtual environment for installing spaCy:</b><br><br>
     <code>python -m venv .env </code>  #create a new virtual env <br>
     <code>source .env/bin/activate </code>  #activate new environment <br>
</div>


#### Install spacy Using pip

<code>pip install -U pip setuptools wheel</code> #update pip, setuptools and wheel<br>

<code>pip install spacy </code>                  # install spaCy   <br>


####  Download and install builtin Trained Pipeline for English Langauge(Small sized pipeline) via spaCy’s download command. 
It takes care of finding the best-matching package compatible with your spaCy installation.<br>

<code>python -m spacy download en_core_web_sm</code><br>


#### 3b. Install spacy using conda

##### Create a new environment using conda  for spaCy  


<div class="alert alert-block alert-info">
    <b>  Create a new Python environment with name "myenv":</b> <br><br>
     <code>conda create --name myenv </code>  #create a new virtual env <br>
     <code>sconda activate myenv </code>  #activate new environment <br>

To relfect new environment in Jupyter Notebook, use following two line of code in this order at Ananconda Prompt:<br>
<code>conda install -c anaconda ipykernel</code><br>
<code>python -m ipykernel install --user --name= myenv</code><br>
</div>


#### Install spaCy in new environment Using conda : 

<code>conda install -c conda-forge spacy</code>  #use conda-forge channel<br>

Download and install builtin Trained Pipeline for English Langauge (Small sized pipeline)<br>

<code>python -m spacy download en_core_web_sm</code>


#### 3c. Updating spaCy

<code>pip install -U spacy</code><br>

##### Once installed, you will need to import spacy in your workspace

<code>import spacy</code><br>


###  3d. Download and install spaCy Trained Pipelines (Models) 

After installing spacy, you are required to install the trained pipelines/models.<br>
Easiest method is to use spacy download command at command prompt, Anaconda prompt or in Jupyter Notebook:<br>
It takes care of finding the best-matching package compatible with your spaCy installation.

<code>python -m spacy download en_core_web_sm</code> #Download and install small English language pipeline.<br>

##### If you are running above line of code to download "en_code_web_sm" from Jupyter Notebook use below code:

<code>!python -m spacy download en_core_web_sm</code><br>

Other larger sized pipelines for English are called en_core_web_md, en_core_web_lg, en_core_web_trf. These differ in size and utility provided. Similarly spaCy has pipelines of different sizes for other languages as well.

#### A spaCy trained Pipeline has components which are trained on labelled data to perform following NLP tasks:

1. Tokenization<br>
2. Parts of speech tagging<br>
3. Name Entity Detection<br>
4. Dependency parsing<br>
5. Matching text with patterns<br>

Once you have downloaded and installed a pipeline e.g., **en_core_web_sm** you can load it in your workspace using **spacy.load()** method. This will return a **Language object** containing all components and data needed to process text. We usually call it **nlp**. Calling the nlp object on a string of text will return a processed **Doc** object.


##### Names and size of some pipelines

**"en_core_web_sm"**   is a small (12MB) English pipeline optimized for CPU. <br>
Its components are: tok2vec, tagger, parser, senter, ner, attribute_ruler, lemmatizer.<br>
 
Other English pipelines : <br>
**en_core_web_md**  - medium sized model (31 MB) <br>
**en_core_web_lg**  - large sized model  (382 MB)<br>
**en_core_web_trf** - English transformer pipeline (483 MB) (roberta-base). <br>
                      Components: transformer, tagger, parser, ner, attribute_ruler, lemmatizer.

<br>
**German Pipelines<br>  **
**de_core_news_sm** - Small German pipeline optimized (13 MB) for CPU. <br>
                      Components: tok2vec, tagger, morphologizer, parser, lemmatizer (trainable_lemmatizer), senter,ner.<br>
**de_core_news_md**<br>
**de_core_news_lg**<br>
**de_core_news_trf**<br>

Once downloaded, you can load langauage specific model with spacy.load() as shown below:<br>

<code><br>
import spacy<br>
# Load small english model<br>
nlp = spacy.load("en_core_web_sm")<br>
</code><br>
<br>
Here nlp is a Langauge object with a lot of built-in text processing functionality. Also known as a **trained pipeline**.



## 4. [Pipelines and Linguistic annotations provided by them](https://spacy.io/usage/spacy-101#annotations)


### 4a.  [Pipelines](https://spacy.io/usage/spacy-101#pipelines)

- When you run **spacy.load()** with the name of a trained language pipeline, a **Language object is created**. This object is usually named as **nlp**. 
- When you call the **nlp** object on your text, it does a series of operation on it. The operations depend on the components in the trained pipeline. This series of operations is known as a **processing pipeline**. 
- Typical components in a trained pipeline are - **Tokenizer, POS tagger, lemmatizer, parser, and Entity Recognizer**. 
- **Tokenizer** splits your text into tokens and create a **Doc** object. Then, the next component takes this **Doc** as input, performs its operation and passes the processed **Doc** to the next component. 

<img src = "https://spacy.io/pipeline-fde48da9b43661abcdf62ab70a546d71.svg" width = 800>
<center><i>Source: "https://spacy.io/usage/spacy-101#pipelines" </i></center>


### 4b.  Linguistic annotations provided by the trained pipelines

- spaCy pipelines predict langauge specific information (linguistic annotations) about your text. All the linguistic annotations are available as Token attributes. 
- After tokenization, spaCy uses the trained components of pipeline in the nlp object to parse the doc object and predict the tags for the tokens in a Doc object. 
- spaCy encodes all text strings to **hash values**. This helps reduce memory usage and improves efficiency. So to print or read the string representation of an attribute, addan underscore _ to its name.  


#### 1. Linguistic annotations: Tokenization
-  First step in a processing pipeline is the tokenization of text. I.e., splitting larger chunks of text into smaller segments as per language specific rules. E.g., sentences into words and punctuation, paragraphs into sentences. 
- Tokenization in spaCy can be done even without a trained pipeline. 

<code><br>
import spacy<br>
nlp = spacy.load("en_core_web_sm")<br>
text = "Hello, My Name is Nimrita"<br>
doc = nlp(text)<br>
for token in doc:<br>
    print(token.text)<br>
</code><br>

#### 2. Linguistic annotations: Part-of-speech tags and dependencies 

spaCy's POS tagger predicts POS label for each token. To see part of speech tag of a token in a Doc object use **token.pos_**.

#### 3. Linguistic annotations: Named Entities 

- Real world objects like a country, a book title, a person, a company that is assigned a name is known as a **"Named Entity"**. 
- spaCy pipeline's **Named Entity Recognizer** component can predict the Named Entity types for text in a document. 
- Named entities are available as the **ents** property of a Doc object.

#### 4. Linguistic annotations: Word vectors and similarity 
- Machine learning/deep learning or statistical models can be trained on numerical data. Therefore, the text strings are converted into numerical vectors known as **Word Vectors** or **word embeddings**. These are multi-dimensional representations of the meaning of a word. Algorithms like **word2vec** are used to generate these word vectors. 

- Small sized trained pipelines for any langauge do not contain word vectors. To use word vectors please download and install medium, large or transformer pipelines. 

<code>!python -m spacy download en_core_web_lg</code><br>

- To see vector for a token use **Token.vector**, you can also see Doc.vector or Span.vector. Doc.vector and Span.vector are the average of their token vectors by default. 

- spaCy computes Similarity among words by comparing their word vectors. It is a floating point number between 0 and 1. spaCy objects Doc, Span, Token and Lexeme come with a .similarity method.

## 5. [Architecture of spaCy](https://spacy.io/usage/spacy-101#architecture)

spaCy architecture has 3 main data structures: 

1. Language Class or the nlp Object - the nlp object is created by spacy.load() or spacy.blank(). 
         It processes the raw text, applies pipeline components to it and returns an annotated Doc object.
2. Doc Object - contains and owns the tokens and their annotations.    
3. [Vocab](https://spacy.io/usage/spacy-101#vocab) Object 
      -  The **Vocab** object in spaCy is a shared Vocabulary that stores the text data in many documents. It uses a hash function to convert text strings, entity labels, part of speech tags etc. into numerical hash values. For example, the word "coffee" has the hash 3197928453018144401. If many documents contain the word "coffee", spaCy hashes the word and stores it only once in the **StringStore**. The StringStore (**doc.vocab.strings**) is a lookup table in which you can look up a string to get its hash, or a hash to get its string.

<div class="alert alert-block alert-info">
<code><br>
import spacy<br>
nlp = spacy.load("en_core_web_sm")<br>
doc = nlp("I love coffee")<br>
print(doc.vocab.strings["coffee"])  # 3197928453018144401<br>
print(doc.vocab.strings[3197928453018144401])  # 'coffee'<br>
</code><br>
</div>
      
- Each entry in Vocabulary is called a **Lexeme**. Vocabulary may not include word text, and can look up it up in an object called the **StringStore** from its hashvalue.  


<img src = "https://d33wubrfki0l68.cloudfront.net/f35bab3fa9d1e91601695dbd6241a607ba11876c/179f6/architecture-415624fc7d149ec03f2736c4aa8b8f3c.svg" width = 500/>

<center><i>Architecture of spaCy.  Source: https://spacy.io/usage/spacy-101</i></center>


### 5a. Other Container objects in spaCy

Besides Langauge, Doc and Vocab there are other important objects which contain text or linguistic annotations.<br>

1. **Token** individual text entities like words, punctuation, spaces, etc. It contains context specific linguistic attributes. 
    Some common lexical attributes of Token object are: is_punct, is_ascii, is_digit, is_bracket, is_quote, is_upper, is_lower,      like_url etc.Linguistic annotations in a Token object are token.pos_, token.dep_ etc. <br>
2. **Lexeme** is an entry in the vocabulary. It contains only structural information of a word, e.g., spelling, nature of characters in the word etc. i.e. the information that won't change based on context. It is unlike a **Token** which contains context dependent linguistic annotaions.<br>
   

<div class="alert alert-block alert-info">
<code><br>
import spacy<br>
nlp = spacy.load("en_core_web_sm")<br>
doc = nlp("I love coffee")<br>
for word in doc:<br>
    lexeme = doc.vocab[word.text]<br>
    print(lexeme.text, lexeme.orth, lexeme.shape_, lexeme.prefix_, lexeme.suffix_,\<br>
            lexeme.is_alpha, lexeme.is_digit, lexeme.is_title, lexeme.lang_)<br>
</code>

Here: <br>
Text: The original text of the lexeme. <br>
Orth: The hash value of the lexeme. <br>
Shape: The abstract word shape of the lexeme. <br>
Prefix: By default, the first letter of the word string. <br>
Suffix: By default, the last three letters of the word string. <br>
is alpha: Does the lexeme consist of alphabetic characters? <br>
is digit: Does the lexeme consist of digits? <br>
</div>

3. **Span** object represents a slice of a **Doc** object.<br>
4. **SpanGroup** object is a named collection of spans belonging to a **Doc**.<br>
5. **DocBin**  is a collection of **Doc** objects used for training data during training of pipeline components.
6. **Example** a collection of training annotations containing two Doc objects: the reference data and the predictions.

There are [other classes](https://spacy.io/usage/spacy-101#architecture-other) in spaCy : Corpus Class, KnowledgeBase, Lookups, MorphAnalys, Morphology, Scorer, StringStore, Vectors, Vocab.  


### 5b. Processing Pipeline in spaCy

- The components of the trained model are applied on the Doc object in a specific order. E.g., Tokenizer is run first of all other components. This series of processing steps forms the processing pipeline. 

- We can add more built-in components or our own custom components to a Langauge object (nlp) using nlp.add_pipe() method. 
<img src = "https://d33wubrfki0l68.cloudfront.net/3ad0582d97663a1272ffc4ccf09f1c5b335b17e9/7f49c/pipeline-fde48da9b43661abcdf62ab70a546d71.svg" width = 500/>
<center><i> Typical Pipeline Components. Source: https://spacy.io/usage/spacy-101 </i></center>
 
There are other built-in spacy components that can be added to a pipeline if your task requires so.

### 5c. Matchers

Matchers help you find and extract matches between patterns or linguistic attributes and the text in a Doc object.    


### 6. [Serialization - Saving your spaCy projects](https://spacy.io/usage/spacy-101#serialization)

You can save your spaCy projects or the customization you do to the spaCy pipeline components by converting them into a byte string. This process is called as serialization. spaCy uses Python's builtin Pickle protocol to serialize your objects and to save them to hard disk or to load them from hard disk. All container classes in spaCy - Language (nlp), Doc, Vocab and StringStore can be serialized.

### 7. [Training (Fine Tuning) the Components of spaCy pipeline](https://spacy.io/usage/spacy-101#training)

- Components of spaCy's pipeline - POS tagger, parser, text categorizer etc. are statistical models that have been trained using supervised machine learning (Backpropagation and Gradient descent). The model weights are provided as a part of the pipeline. However, some of these components can be fine tuned with your own data or your own annotations. For this, you can use spacy train() method with configuration settings of your choice. 

#### Trainable components

- **Pipe** class in spaCy allows you to create your own components/models to make predictions of your choice on the Doc objects. These components can be tained using train(). You can use these components as part of the spaCy pipeline. 

#### Training config and lifecycle

- Pipeline contains **Training config file** called as **config.cfg** that include all settings and hyperparameters for training your pipeline. You can call train() method and pass the training configuration by writing it in the config.cfg file. There is no need to provide the training parameters on the command line. 

### 8. [Language data](https://spacy.io/usage/spacy-101#language-data)

The **lang** module of spaCy contains language specific data that is organized in simple Python files. It includes data like which words are very common, what are the exceptions or special cases of word usage in that language. The root directory contains shared language data that has the rules for basic punctuation , emoji, emoticons, single letter abbreviations that can be shared across languages. Subdirectories/submodules contain language specific rules. 

<div class="alert alert-block alert-info">
<code><br>
from spacy.lang.en import English<br>
from spacy.lang.de import German<br>
nlp_en = English()  # Includes English data<br>
nlp_de = German()  # Includes German data<br>
</code><br>
</div>

### Code Demo



```python
# import spacy into your workspace and print the version of spacy you have
import spacy
print(spacy.__version__)
```

    3.3.1
    


```python
## Load small English pipeline and apply it to your text. Print the tokens of your text
import spacy
nlp = spacy.load("en_core_web_sm")
text = "Hello, My Name is Nimrita"
doc = nlp(text)
for token in doc:
    print(token.text)
```

    Hello
    ,
    My
    Name
    is
    Nimrita
    

#### [Lemmatization](https://spacy.io/usage/linguistic-features)

Lemmatization is the process of identifying the root word from a group of related words. 
E.g. 'ate', 'eat', 'eating' all are derived from the word 'eat'. 
Buy, bought, buying, buyer all are derived from the word 'buy'. 
This helps in correct interpretation of meaning of the text and reduces the overall amount of text to tbe processed by NLP models. **Token.lemma_** attribute helps you identigy the lemma of a token in spaCy.


#### [Dependency Parsing](https://spacy.io/usage/linguistic-features#dependency-parse)

[The Stanford NLP Group](https://nlp.stanford.edu/software/nndep.html) defines the [task of a dependency parser](https://nlp.stanford.edu/software/nndep.html) as follows:<br>

A dependency parser analyzes the grammatical structure of a sentence, establishing relationships between "head" words and words which modify those heads. In the figure below we see a dependency parse of a short sentence. The arrow from the word "moving" to the word "faster" indicates that "faster" modifies "moving", and the label "advmod" assigned to the arrow describes the  nature of the dependency as adverb modifier.

<img src = "https://nlp.stanford.edu/static/img/nndep-example.png" width="600"/>
<center><i>Source Credits: https://nlp.stanford.edu/static/img/nndep-example.png</i></center>

**token.dep_** attribute can be used to know the dependency relationship of a token to other tokens in spaCy. 


```python
# Print token text, lemma, pos tag, syntactic dependency, shape, and some lexical attributes of tokens 
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Google Inc. is buying a U.K. company for $110 million dollars")
for token in doc:
    print(f"TokenText: {token.text:9} Lemma: {token.lemma_:<8} \
          POS: {token.pos_:6}  Dep: {token.dep_:9} Shape: {token.shape_:6} IsAlphabet: {token.is_alpha:<2} Stopword: {token.is_stop:<2}") 
    
```

    TokenText: Google    Lemma: Google             POS: PROPN   Dep: compound  Shape: Xxxxx  IsAlphabet: 1  Stopword: 0 
    TokenText: Inc.      Lemma: Inc.               POS: PROPN   Dep: nsubj     Shape: Xxx.   IsAlphabet: 0  Stopword: 0 
    TokenText: is        Lemma: be                 POS: AUX     Dep: aux       Shape: xx     IsAlphabet: 1  Stopword: 1 
    TokenText: buying    Lemma: buy                POS: VERB    Dep: ROOT      Shape: xxxx   IsAlphabet: 1  Stopword: 0 
    TokenText: a         Lemma: a                  POS: DET     Dep: det       Shape: x      IsAlphabet: 1  Stopword: 1 
    TokenText: U.K.      Lemma: U.K.               POS: PROPN   Dep: compound  Shape: X.X.   IsAlphabet: 0  Stopword: 0 
    TokenText: company   Lemma: company            POS: NOUN    Dep: dobj      Shape: xxxx   IsAlphabet: 1  Stopword: 0 
    TokenText: for       Lemma: for                POS: ADP     Dep: prep      Shape: xxx    IsAlphabet: 1  Stopword: 1 
    TokenText: $         Lemma: $                  POS: SYM     Dep: quantmod  Shape: $      IsAlphabet: 0  Stopword: 0 
    TokenText: 110       Lemma: 110                POS: NUM     Dep: compound  Shape: ddd    IsAlphabet: 0  Stopword: 0 
    TokenText: million   Lemma: million            POS: NUM     Dep: nummod    Shape: xxxx   IsAlphabet: 1  Stopword: 0 
    TokenText: dollars   Lemma: dollar             POS: NOUN    Dep: pobj      Shape: xxxx   IsAlphabet: 1  Stopword: 0 
    

#### spacy.explain() to understand the tags you don't follow


```python
#Using spacy.explain() function , you can know the explanation or full-form of any Term that you may not know of.
spacy.explain('quantmod')
```




    'modifier of quantifier'



#### Print all Named Entities in text
Named Entities in text are the words that represent real world entities that have a name. For example, a person, a country, a company. 


```python
# Print all Named Entities in text
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Google Inc. is buying a U.K. company for $110 million dollars")

for ent in doc.ents:
    print(f"Entity Text: {ent.text:23} Entity Label: {ent.label_:10}")

```

    Entity Text: Google Inc.             Entity Label: ORG       
    Entity Text: U.K.                    Entity Label: GPE       
    Entity Text: $110 million dollars    Entity Label: MONEY     
    

### Word Vectors and Similarity 
Word vectors are numerical representations of the meaning of words. Since statistical models work with numbers so all text needs to be represented in the form of numerical vectors. Common algorithms like Word2Vec or Token2vec generate such word embeddings or vector representations from text. Vectors for Doc are average of the vectors for vectors of its tokens.


```python
## Print word vectors and similarity between words
# Small pipelines dont contain word vectors, so we have downloaded and installed medium size English pipeline for this code
# Similarity is obtained using the .similarity() method that can be applied on Doc, Span, View objects.

import spacy
nlp = spacy.load("en_core_web_md")

text1 = "dog cat banana afskfsd"
doc1 = nlp(text1)
for token in doc1:
    print(f"TokenText: {token.text:8} HasVector: {token.has_vector:2} L2 Norm: {token.vector_norm:18} OOV: {token.is_oov}")
    print(f"Token Vector \n: {token.vector}")

    
text2 = "i have a dog and a cat"
doc2 = nlp(text2)

text3 = "i love rain"
doc3 = nlp(text3)

text4 = "cat banana"
doc4 = nlp(text4)

print(f'\n\nSimilarity of doc1 with doc2: {doc1.similarity(doc2)} ')

print(f'\n\nSimilarity of doc1 with doc3: {doc1.similarity(doc3)} ')

print(f'\n\nSimilarity of doc1 with doc4: {doc1.similarity(doc4)} ')
```

    TokenText: dog      HasVector:  1 L2 Norm:  7.443447113037109 OOV: False
    Token Vector 
    : [-0.72483    0.42538    0.025489  -0.39807    0.037463  -0.29811
     -0.28279    0.29333    0.57775    1.2205    -0.27903    0.80879
     -0.71291    0.045808  -0.46751    0.55944    0.42745    0.58238
      0.20854   -0.42718   -0.40284   -0.048941   0.1149    -0.6963
     -0.03338    0.052596  -0.22572   -0.35996    0.47961   -0.38386
     -0.73837    0.1718     0.52188    0.45584   -0.026621   0.48831
      0.67996   -0.73345   -0.27078    0.41739    0.1947     0.27389
     -0.70931   -0.45317   -0.22574   -0.12617    0.03268    0.142
      0.53923   -0.61285   -0.5322     0.19479    0.13889   -0.020284
      0.088162   0.85337    0.039407   0.11529   -0.42646    0.74832
      0.34421   -0.59462    0.0040537  0.027203  -0.063394   0.26538
      0.34757    0.21395   -0.39799   -0.027067  -0.36132    0.31979
      0.55813   -0.5652     0.55382    0.03928   -0.26933   -0.14705
      0.74032   -0.50566    0.023765   0.62273   -0.79388   -0.25165
      0.11992   -0.43056    1.0614     0.58571    0.8856    -0.056054
      0.055826   0.30485    0.64639   -0.43831   -0.45706    0.036471
     -0.3466    -0.56219    0.28105   -0.33758   -0.041398   0.22171
      0.05262    0.18113    0.65646   -0.56217    0.038915  -0.30335
      0.05051   -0.2354     0.3233     0.31744    0.52453   -0.47154
      0.13152   -0.15104    0.14265   -0.20747    0.060413  -0.030342
     -0.092883   0.80421   -0.12497   -0.56199    0.29128   -0.22488
      0.30282   -0.0045144 -0.12305    0.20396   -0.32202   -0.11409
     -0.37613    0.40457    0.21461    0.25741   -0.36489    0.94135
      0.42725    0.022925  -1.8699    -0.76035    0.73771    0.36998
      0.50214   -0.30617   -0.26526    0.86573    0.3808     0.14754
      0.29932   -0.078863  -0.28992   -0.064636  -0.68914    0.19527
     -0.56368    0.26251   -0.52171   -1.0703     0.42478   -0.0067289
     -0.28591   -0.77831    0.049342   0.66675   -0.077419  -0.19226
      0.12721   -0.18844    0.13647    0.38804    0.21917   -0.24192
     -0.13465    0.23119   -0.43197    0.48302    0.3598     1.128
      0.019894  -0.10861   -0.13515   -0.34137   -0.36379    0.080616
      0.28682   -0.045819  -0.12114   -0.44835   -0.054611  -0.10362
      0.010954  -0.60063   -0.46665    0.15115   -0.31815   -0.58903
      1.1325     0.04406   -0.92863    0.3399    -0.03463   -0.40474
      0.17245   -0.19983   -0.095982  -0.074758   0.57472    0.25455
     -0.20387    0.055758  -0.65017    0.72629   -0.51083    0.11196
      0.44724    0.16157   -0.34571    0.19227   -0.063871   0.0057351
      0.48703   -0.53762   -0.73398   -0.11488    0.073723   0.58191
      0.33192   -0.13303   -0.3478    -0.022676  -0.32494   -0.26496
      0.56275    0.098558  -0.16671   -0.40481    0.55477   -0.58692
     -0.60433   -0.4227    -0.53712    0.2994     0.11339   -0.3154
     -0.28685    0.43999    0.013623   0.011139  -0.47734   -0.01492
      0.52524    0.53583    0.36626    0.23119   -0.1386     0.35374
     -0.27448    0.066183   0.6224    -0.24851   -0.36066    0.009084
     -0.58148    0.24371    0.29944   -0.025314  -0.73222    0.33236
     -0.40339    0.82624    0.006984   0.26737   -0.27695   -0.09713
     -0.015736   0.1024    -0.026831  -0.26293    0.31401    0.01051
     -0.048451  -0.74571    0.75827    0.67771    0.054738  -0.23325
      0.17996   -0.206      0.019095  -0.34283   -0.58602    0.0095634
     -0.085052   0.83312    0.31978    0.050317  -0.23159    0.28165  ]
    TokenText: cat      HasVector:  1 L2 Norm:  7.443447113037109 OOV: False
    Token Vector 
    : [-0.72483    0.42538    0.025489  -0.39807    0.037463  -0.29811
     -0.28279    0.29333    0.57775    1.2205    -0.27903    0.80879
     -0.71291    0.045808  -0.46751    0.55944    0.42745    0.58238
      0.20854   -0.42718   -0.40284   -0.048941   0.1149    -0.6963
     -0.03338    0.052596  -0.22572   -0.35996    0.47961   -0.38386
     -0.73837    0.1718     0.52188    0.45584   -0.026621   0.48831
      0.67996   -0.73345   -0.27078    0.41739    0.1947     0.27389
     -0.70931   -0.45317   -0.22574   -0.12617    0.03268    0.142
      0.53923   -0.61285   -0.5322     0.19479    0.13889   -0.020284
      0.088162   0.85337    0.039407   0.11529   -0.42646    0.74832
      0.34421   -0.59462    0.0040537  0.027203  -0.063394   0.26538
      0.34757    0.21395   -0.39799   -0.027067  -0.36132    0.31979
      0.55813   -0.5652     0.55382    0.03928   -0.26933   -0.14705
      0.74032   -0.50566    0.023765   0.62273   -0.79388   -0.25165
      0.11992   -0.43056    1.0614     0.58571    0.8856    -0.056054
      0.055826   0.30485    0.64639   -0.43831   -0.45706    0.036471
     -0.3466    -0.56219    0.28105   -0.33758   -0.041398   0.22171
      0.05262    0.18113    0.65646   -0.56217    0.038915  -0.30335
      0.05051   -0.2354     0.3233     0.31744    0.52453   -0.47154
      0.13152   -0.15104    0.14265   -0.20747    0.060413  -0.030342
     -0.092883   0.80421   -0.12497   -0.56199    0.29128   -0.22488
      0.30282   -0.0045144 -0.12305    0.20396   -0.32202   -0.11409
     -0.37613    0.40457    0.21461    0.25741   -0.36489    0.94135
      0.42725    0.022925  -1.8699    -0.76035    0.73771    0.36998
      0.50214   -0.30617   -0.26526    0.86573    0.3808     0.14754
      0.29932   -0.078863  -0.28992   -0.064636  -0.68914    0.19527
     -0.56368    0.26251   -0.52171   -1.0703     0.42478   -0.0067289
     -0.28591   -0.77831    0.049342   0.66675   -0.077419  -0.19226
      0.12721   -0.18844    0.13647    0.38804    0.21917   -0.24192
     -0.13465    0.23119   -0.43197    0.48302    0.3598     1.128
      0.019894  -0.10861   -0.13515   -0.34137   -0.36379    0.080616
      0.28682   -0.045819  -0.12114   -0.44835   -0.054611  -0.10362
      0.010954  -0.60063   -0.46665    0.15115   -0.31815   -0.58903
      1.1325     0.04406   -0.92863    0.3399    -0.03463   -0.40474
      0.17245   -0.19983   -0.095982  -0.074758   0.57472    0.25455
     -0.20387    0.055758  -0.65017    0.72629   -0.51083    0.11196
      0.44724    0.16157   -0.34571    0.19227   -0.063871   0.0057351
      0.48703   -0.53762   -0.73398   -0.11488    0.073723   0.58191
      0.33192   -0.13303   -0.3478    -0.022676  -0.32494   -0.26496
      0.56275    0.098558  -0.16671   -0.40481    0.55477   -0.58692
     -0.60433   -0.4227    -0.53712    0.2994     0.11339   -0.3154
     -0.28685    0.43999    0.013623   0.011139  -0.47734   -0.01492
      0.52524    0.53583    0.36626    0.23119   -0.1386     0.35374
     -0.27448    0.066183   0.6224    -0.24851   -0.36066    0.009084
     -0.58148    0.24371    0.29944   -0.025314  -0.73222    0.33236
     -0.40339    0.82624    0.006984   0.26737   -0.27695   -0.09713
     -0.015736   0.1024    -0.026831  -0.26293    0.31401    0.01051
     -0.048451  -0.74571    0.75827    0.67771    0.054738  -0.23325
      0.17996   -0.206      0.019095  -0.34283   -0.58602    0.0095634
     -0.085052   0.83312    0.31978    0.050317  -0.23159    0.28165  ]
    TokenText: banana   HasVector:  1 L2 Norm:   6.89589786529541 OOV: False
    Token Vector 
    : [-0.6334     0.18981   -0.53544   -0.52658   -0.30001    0.30559
     -0.49303    0.14636    0.012273   0.96802    0.0040354  0.25234
     -0.29864   -0.014646  -0.24905   -0.67125   -0.053366   0.59426
     -0.068034   0.10315    0.66759    0.024617  -0.37548    0.52557
      0.054449  -0.36748   -0.28013    0.090898  -0.025687  -0.5947
     -0.24269    0.28603    0.686      0.29737    0.30422    0.69032
      0.042784   0.023701  -0.57165    0.70581   -0.20813   -0.03204
     -0.12494   -0.42933    0.31271    0.30352    0.09421   -0.15493
      0.071356   0.15022   -0.41792    0.066394  -0.034546  -0.45772
      0.57177   -0.82755   -0.27885    0.71801   -0.12425    0.18551
      0.41342   -0.53997    0.55864   -0.015805  -0.1074    -0.29981
     -0.17271    0.27066    0.043996   0.60107   -0.353      0.6831
      0.20703    0.12068    0.24852   -0.15605    0.25812    0.007004
     -0.10741   -0.097053   0.085628   0.096307   0.20857   -0.23338
     -0.077905  -0.030906   1.0494     0.55368   -0.10703    0.052234
      0.43407   -0.13926    0.38115    0.021104  -0.40922    0.35972
     -0.28898    0.30618    0.060807  -0.023517   0.58193   -0.3098
      0.21013   -0.15557   -0.56913   -1.1364     0.36598   -0.032666
      1.1926     0.12825   -0.090486  -0.47965   -0.61164   -0.16484
     -0.41134    0.19925    0.059183  -0.20842    0.45223    0.27697
     -0.20745    0.025404  -0.28874    0.040478  -0.22275   -0.43323
      0.76957   -0.054327  -0.35213   -0.30842   -0.48791   -0.35564
      0.19813   -0.094767  -0.50918    0.18763   -0.087555   0.37709
     -0.1322    -0.096913  -1.9102     0.55813    0.27391   -0.077744
     -0.43933   -0.10367   -0.24408    0.41869    0.11659    0.27454
      0.81021   -0.11006    0.43131    0.29095   -0.49548   -0.31958
     -0.072506   0.020286   0.2179     0.22032   -0.29212    0.75639
      0.13598    0.019736  -0.83104    0.22836   -0.28669   -1.0529
      0.052771   0.41266    0.50149    0.5323     0.51573   -0.31806
     -0.4619     0.21739   -0.43584   -0.41382    0.042237  -0.57179
      0.067623  -0.27854    0.090044   0.20633    0.024678  -0.57703
     -0.020183  -0.53147   -0.37548   -0.12795   -0.093662  -0.0061183
      0.20221   -0.62296   -0.29746    0.26935    0.59009   -0.50382
     -0.69757    0.20157   -0.33592   -0.45766    0.14061    0.22982
      0.044046   0.26386    0.02942    0.34095    1.1496    -0.15555
     -0.064071   0.30139    0.024211  -0.63515   -0.73347   -0.10346
     -0.22637   -0.056392  -0.16735   -0.097331  -0.19206   -0.18866
      0.15116   -0.038048   0.70205    0.11586   -0.14813    0.0095166
     -0.33804   -0.10158   -0.23829   -0.22759    0.092504  -0.29839
     -0.39721    0.26092    0.34594   -0.47396   -0.25725   -0.19257
     -0.53071    0.1692    -0.47252   -0.17333   -0.40505    0.046446
     -0.04473    0.33555   -0.5693     0.31591   -0.21167   -0.31298
     -0.45923   -0.083091   0.086822   0.01264    0.43779    0.12651
      0.30156    0.022061   0.26549   -0.29455   -0.14838    0.033692
     -0.37346   -0.075343  -0.56498   -0.24207   -0.69351   -0.20277
     -0.0081185  0.030971   0.53615   -0.16613   -0.84087    0.74661
      0.029132   0.46936   -0.49755    0.40954   -0.022558   0.21497
     -0.049528  -0.039799   0.46165    0.26456    0.32985   -0.04219
     -0.099599  -0.17312   -0.476     -0.019048  -0.41888   -0.2685
     -0.65281    0.068773  -0.23881   -1.1784     0.25504    0.61171  ]
    TokenText: afskfsd  HasVector:  0 L2 Norm:                0.0 OOV: True
    Token Vector 
    : [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    
    
    Similarity of doc1 with doc2: 0.7071206680444951 
    
    
    Similarity of doc1 with doc3: 0.3090511739579275 
    
    
    Similarity of doc1 with doc4: 0.9685581106439412 
    

**'dog' and 'cat'** are common English words, hence these words and their vectors are included in pipeline's vocabulary. But **"afskfsd"** is not common and hence **OOV : out-of-vocabulary**, so its vector representation consists of 300 dimensions of 0, which means that this is not a valid existing word in the English langauge. 

#### Text-Cleaning / Preprocessing with spaCy
- Text cleaning or preprocessing involves removing the parts of text that do not add value to the meaning to the text. These are the stopwords, punctuation or spaces.  
- **Stopwords** are the words like “a”, ” the”, “was”, "it" that occur very frequently in English lanaguage text and  do not add value to its meaning. 
- spaCy tokens have attributes that tell you if a word is noise. E.g., token.is_stop, token.is_punct, token.is_space 


```python
import spacy

#create some text
text = """I love to read a book on a beach. The breeze is relaxing!"""

# Here we have not loaded a trained pipeline, this is spacy.blank() using simple English langauge Tokenizer.
nlp = spacy.blank('en')

#apply nlp object to text. it creates a Doc with tokens and some attributes

doc = nlp(text)

# Print the tokens and whether they are stopwords or punctuation
for token in doc:
    print(f'Token Text: {token.text:10} Stop word?: {token.is_stop:2} Punctuation?: {token.is_punct:2}')
    
# Removing StopWords and punctuations
doc_cleaned = [token for token in doc if not token.is_stop and not token.is_punct]

print('\n\nDoc after removing stopwords and punctuation')
for token in doc_cleaned:
    print(token.text)
```

    Token Text: I          Stop word?:  1 Punctuation?:  0
    Token Text: love       Stop word?:  0 Punctuation?:  0
    Token Text: to         Stop word?:  1 Punctuation?:  0
    Token Text: read       Stop word?:  0 Punctuation?:  0
    Token Text: a          Stop word?:  1 Punctuation?:  0
    Token Text: book       Stop word?:  0 Punctuation?:  0
    Token Text: on         Stop word?:  1 Punctuation?:  0
    Token Text: a          Stop word?:  1 Punctuation?:  0
    Token Text: beach      Stop word?:  0 Punctuation?:  0
    Token Text: .          Stop word?:  0 Punctuation?:  1
    Token Text: The        Stop word?:  1 Punctuation?:  0
    Token Text: breeze     Stop word?:  0 Punctuation?:  0
    Token Text: is         Stop word?:  1 Punctuation?:  0
    Token Text: relaxing   Stop word?:  0 Punctuation?:  0
    Token Text: !          Stop word?:  0 Punctuation?:  1
    
    
    Doc after removing stopwords and punctuation
    love
    read
    book
    beach
    breeze
    relaxing
    

#### spaCy uses POS tag == 'X' for words like etc, i.e. indicating that they are slang/junk words


```python
# Identifying all POS tags for tokens in a piece of text.
import spacy
nlp = spacy.load("en_core_web_sm")

text= """He 26 books. $ 10.5 plus 10% tax etc etc. """

# Pass the Text to Model
doc = nlp(text)

print('Number of tokens before preprocessing: ', len(doc))

print("\nNow preprocessing to remove words with POS == X, stopwords, and punctuation...")

cleaned_doc = [token for token in doc if not token.pos_=='X' and not token.is_stop and not token.is_punct]

print('\nNo. of tokens after removing stopwords and punctuation: ', len(cleaned_doc))

print('\nCleaned Doc:')
for token in cleaned_doc:
    print(f'Token Text:{token.text:10}  POS: {token.pos_:8}')
    

```

    Number of tokens before preprocessing:  13
    
    Now preprocessing to remove words with POS == X, stopwords, and punctuation...
    
    No. of tokens after removing stopwords and punctuation:  7
    
    Cleaned Doc:
    Token Text:26          POS: NUM     
    Token Text:books       POS: NOUN    
    Token Text:$           POS: SYM     
    Token Text:10.5        POS: NUM     
    Token Text:plus        POS: CCONJ   
    Token Text:10          POS: NUM     
    Token Text:tax         POS: NOUN    
    

#### Printing the tokens which are like numbers, percentage numbers, email addresses or URLs


```python
#Printing the tokens which are like numbers or email addresses

import spacy

nlp = spacy.load("en_core_web_sm")

text= """Covid19 made 2020 a catastropic year, 10% people who got infected died. 
       All casualties were reported to jhucasualties@ada.com. www.jhu.org has the details."""

doc=nlp(text)

for token in doc:
    if token.like_num:
        print("A number:", token)
        index_of_next_token=token.i + 1
        next_token = doc[index_of_next_token]
        if next_token.text == '%':
            print("A percentage number:", token.text)
    if token.like_email:
        print("An email address: ", token.text) 
    if token.like_url:
        print("A URL: ", token.text) 
```

    A number: 2020
    A number: 10
    A percentage number: 10
    An email address:  jhucasualties@ada.com
    A URL:  www.jhu.org
    

#### Morphology 
In a natural language, a lemma (root form) of a word is inflected (modified/combined) with one or more morphological features to create many froms of the root form. Here are some examples:

root form : read
inflected forms : reading, read

Morphological features are stored in the MorphAnalysis under Token.morph, which allows you to access individual morphological features.


```python
import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("I was reading the newspaper.")
token = doc[0]  # 'I'
print(token.morph)  

doc = nlp("She is reading the paper.")
token = doc[2]  # 'reading'
print(token.morph)  

```

    Case=Nom|Number=Sing|Person=1|PronType=Prs
    Aspect=Prog|Tense=Pres|VerbForm=Part
    

### Print hash and the corresponding token text from StringStore class 


```python
# Print hash and the corresponfing token text

import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("Sherlock Holmes solved the murder mystery like a piece of cake")

# Look up the hash for the word "murder" in nlp.vocab.strings
hash = nlp.vocab.strings["murder"]
print("hash of the word 'murder' is {}".format(hash))

# Look up the word using a hash value in nlp.vocab.strings 
word = nlp.vocab.strings[hash]
print("Word corresponding to the hash {} is '{}''".format(hash, word))
```

    hash of the word 'murder' is 730091238969817242
    Word corresponding to the hash 730091238969817242 is 'murder''
    

#### [Named Entity Recognition](https://spacy.io/usage/linguistic-features#named-entities-101)

spaCy's trained language pipeline models can predict various types of named entities in a document.
all the named entities are available in **doc.ents** property. 

The common Named Entity categories supported by spacy are :<br>

PERSON : Names of people<br>
GPE : Geo physical entity - like counties, cities, states<br>
ORG : Organizations or companies<br>
WORK_OF_ART : Titles of books, fimls,songs and other arts<br>
PRODUCT : Products such as vehicles, food items ,furniture and so on<br>
EVENT : Historical events like wars, disasters ,etc <br>
LANGUAGE : All the recognized languages across the globe<br>


```python
### See all entity types in  specific langauge model
import spacy

nlp = spacy.load("en_core_web_sm")
nlp.get_pipe("ner").labels
```




    ('CARDINAL',
     'DATE',
     'EVENT',
     'FAC',
     'GPE',
     'LANGUAGE',
     'LAW',
     'LOC',
     'MONEY',
     'NORP',
     'ORDINAL',
     'ORG',
     'PERCENT',
     'PERSON',
     'PRODUCT',
     'QUANTITY',
     'TIME',
     'WORK_OF_ART')




```python
# Identofy all the entities that are ORG in given text
import spacy

nlp = spacy.load("en_core_web_sm")

text = """ This is an important industrial region of the country. 
           It houses the offices of Google, Samsung, Microsoft etc.
           Mr. John works in Walmart ."""

# creating spacy doc
doc = nlp(text)

# Printing the named entities
print('All entities in this doc', doc.ents)


print("\nAll the companies: ")
for entity in doc.ents:
    if entity.label_=='ORG':
        print(entity.text)

        
print("\n\nAll the people: ")        
for entity in doc.ents:
    if entity.label_== 'PERSON':
        print(entity.text)
```

    All entities in this doc (Google, Samsung, Microsoft, John, Walmart)
    
    All the companies: 
    Google
    Samsung
    Microsoft
    Walmart
    
    
    All the people: 
    John
    

#### [Masking certain entity types and tokens representing personal information for privacy](https://www.machinelearningplus.com/spacy-tutorial-nlp/#application2automaticallymaskingentities)
Here in the code below, we mask the token that have entity type PERSON , ORG, GPE , PRODUCT or EVENT, and tokens that are numbers or email addresses.


```python
#Source: https://www.machinelearningplus.com/spacy-tutorial-nlp/#application2automaticallymaskingentities

import spacy

nlp = spacy.load('en_core_web_sm')

text = """Hi, my name is Dr. John. I live in Italy. I work at Google. My phone number is 123456. My email is abcd@abcd.com"""

# creating spacy doc
doc = nlp(text)

print("Original Doc:\n [{}]".format(doc))

print('\nAll entities in this doc', doc.ents)

print("\nIdentified named entities: ")
for ent in doc.ents:
    print("Entity: {:<10} Entity Label: {:<4}".format(ent.text, ent.label))

def find_personaldata(token):
    if token.ent_type_ == 'PERSON' or token.ent_type_== 'ORG' or token.ent_type_== 'GPE' or token.like_num or token.like_email:
        return 'UNKNOWN'
    return token.text

def mask_personaldata(doc):
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            retokenizer.merge(ent)
    tokens = map(find_personaldata, doc)
    return ' '.join(tokens)

print('\nMasked Personal Identification Information')
mask_personaldata(doc)    

```

    Original Doc:
     [Hi, my name is Dr. John. I live in Italy. I work at Google. My phone number is 123456. My email is abcd@abcd.com]
    
    All entities in this doc (John, Italy, Google, 123456)
    
    Identified named entities: 
    Entity: John       Entity Label: 380 
    Entity: Italy      Entity Label: 384 
    Entity: Google     Entity Label: 383 
    Entity: 123456     Entity Label: 391 
    
    Masked Personal Identification Information
    




    'Hi , my name is Dr. UNKNOWN . I live in UNKNOWN . I work at UNKNOWN . My phone number is UNKNOWN . My email is UNKNOWN'



#### Rule based Matching

- spaCy allows you to find and extract tokens and Docs that match a specific text pattern or have certain lexical attributes through use of a rule based matching engine **Matcher**. <br>
- For example, find the word "duck" only if it's a verb, not a noun.<br>

##### Example rules for pattern matching in the form of lists of dictionaries, one per token:<br>

- Match exact token texts    : [{"TEXT": "iPhone"}, {"TEXT": "X"}]<br>
- Match lexical attributes   : [{"LOWER": "iphone"}, {"LOWER": "x"}]<br>
- Match any token attributes : [{"LEMMA": "buy"}, {"POS": "NOUN"}] , [{"TEXT": "iPhone"}, {"TEXT": "X"}]<br>

##### The procedure to implement a token matcher is:

1. Initialize a Matcher object<br>
2. Define the pattern you want to match<br>
3. Add the pattern to the matcher<br>
4. Pass the text to the matcher to extract the matching positions.<br>


```python
#Source: https://spacy.io/usage/rule-based-matching
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
# Add match ID "HelloWorld" with no callback and one pattern
pattern = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
matcher.add("mypattern", [pattern])

text = "Hello, world! Hello world! Hello, world"

doc = nlp(text)
matches = matcher(doc)

for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]  # match_id is the hash value of matched string, here we get the string representation of this hash value
    span = doc[start:end]  # The matched span within the doc
    print(start, end, span.text) # print start token position,  end token position and matches span of each match
```

    0 3 Hello, world
    7 10 Hello, world
    

####  Visualizers - displaCy and displaCy ENT(https://spacy.io/usage/visualizers)

- You can easily visualize dependencies identified by a langauge models in your text using two spaCy visualizers - [displacy](https://explosion.ai/demos/displacy) and displaCy Named Entity Visualizer [displaCy ENT](https://explosion.ai/demos/displacy-ent). Both are part of the core spaCy library. 
- displaCy can take a single Doc or a list of Doc objects as its first argument, it can return the markup that can be rendered in jupyter notebook or exported. 
- displaCy.serve creates a simple web server to see your visualization in a browser. 
  


```python
# Displaying dependency relations between tokens
import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
text = 'I love to read'
doc = nlp(text)
displacy.render(doc,style='dep',jupyter=True)
```


<span class="tex2jax_ignore"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:lang="en" id="e725abc1fa5941d2a3b0845eb8183079-0" class="displacy" width="750" height="312.0" direction="ltr" style="max-width: none; height: 312.0px; color: #000000; background: #ffffff; font-family: Arial; direction: ltr">
<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="50">I</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="50">PRON</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="225">love</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="225">VERB</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="400">to</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="400">PART</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="575">read</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="575">VERB</tspan>
</text>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-e725abc1fa5941d2a3b0845eb8183079-0-0" stroke-width="2px" d="M70,177.0 C70,89.5 220.0,89.5 220.0,177.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-e725abc1fa5941d2a3b0845eb8183079-0-0" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">nsubj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M70,179.0 L62,167.0 78,167.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-e725abc1fa5941d2a3b0845eb8183079-0-1" stroke-width="2px" d="M420,177.0 C420,89.5 570.0,89.5 570.0,177.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-e725abc1fa5941d2a3b0845eb8183079-0-1" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">aux</textPath>
    </text>
    <path class="displacy-arrowhead" d="M420,179.0 L412,167.0 428,167.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-e725abc1fa5941d2a3b0845eb8183079-0-2" stroke-width="2px" d="M245,177.0 C245,2.0 575.0,2.0 575.0,177.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-e725abc1fa5941d2a3b0845eb8183079-0-2" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">xcomp</textPath>
    </text>
    <path class="displacy-arrowhead" d="M575.0,179.0 L583.0,167.0 567.0,167.0" fill="currentColor"/>
</g>
</svg></span>


#### displacy.serve visualizing [dependency parsing](https://spacy.io/usage/visualizers#dep)


```python
#visualizing dependency parsing
import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")

text = """We live in adjacent houses."""
doc = nlp(text)
sentence_spans = list(doc.sents)
displacy.serve(sentence_spans, style="dep")
```


<span class="tex2jax_ignore"><!DOCTYPE html>
<html lang="en">
    <head>
        <title>displaCy</title>
    </head>

    <body style="font-size: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; padding: 4rem 2rem; direction: ltr">
<figure style="margin-bottom: 6rem">
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xml:lang="en" id="2e34b69b197a46a3b5439f432730b956-0" class="displacy" width="925" height="312.0" direction="ltr" style="max-width: none; height: 312.0px; color: #000000; background: #ffffff; font-family: Arial; direction: ltr">
<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="50">We</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="50">PRON</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="225">live</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="225">VERB</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="400">in</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="400">ADP</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="575">adjacent</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="575">ADJ</tspan>
</text>

<text class="displacy-token" fill="currentColor" text-anchor="middle" y="222.0">
    <tspan class="displacy-word" fill="currentColor" x="750">houses.</tspan>
    <tspan class="displacy-tag" dy="2em" fill="currentColor" x="750">NOUN</tspan>
</text>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-2e34b69b197a46a3b5439f432730b956-0-0" stroke-width="2px" d="M70,177.0 C70,89.5 220.0,89.5 220.0,177.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-2e34b69b197a46a3b5439f432730b956-0-0" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">nsubj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M70,179.0 L62,167.0 78,167.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-2e34b69b197a46a3b5439f432730b956-0-1" stroke-width="2px" d="M245,177.0 C245,89.5 395.0,89.5 395.0,177.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-2e34b69b197a46a3b5439f432730b956-0-1" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">prep</textPath>
    </text>
    <path class="displacy-arrowhead" d="M395.0,179.0 L403.0,167.0 387.0,167.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-2e34b69b197a46a3b5439f432730b956-0-2" stroke-width="2px" d="M595,177.0 C595,89.5 745.0,89.5 745.0,177.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-2e34b69b197a46a3b5439f432730b956-0-2" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">amod</textPath>
    </text>
    <path class="displacy-arrowhead" d="M595,179.0 L587,167.0 603,167.0" fill="currentColor"/>
</g>

<g class="displacy-arrow">
    <path class="displacy-arc" id="arrow-2e34b69b197a46a3b5439f432730b956-0-3" stroke-width="2px" d="M420,177.0 C420,2.0 750.0,2.0 750.0,177.0" fill="none" stroke="currentColor"/>
    <text dy="1.25em" style="font-size: 0.8em; letter-spacing: 1px">
        <textPath xlink:href="#arrow-2e34b69b197a46a3b5439f432730b956-0-3" class="displacy-label" startOffset="50%" side="left" fill="currentColor" text-anchor="middle">pobj</textPath>
    </text>
    <path class="displacy-arrowhead" d="M750.0,179.0 L758.0,167.0 742.0,167.0" fill="currentColor"/>
</g>
</svg>
</figure>
</body>
</html></span>


    
    Using the 'dep' visualizer
    Serving on http://0.0.0.0:5000 ...
    
    Shutting down server on port 5000.
    


```python
# Visualizing named entities
import spacy
from spacy import displacy

text = """Sebastian Thrun started working at Google Inc. in 2007"""

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
displacy.serve(doc, style="ent")
```


<span class="tex2jax_ignore"><!DOCTYPE html>
<html lang="en">
    <head>
        <title>displaCy</title>
    </head>

    <body style="font-size: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; padding: 4rem 2rem; direction: ltr">
<figure style="margin-bottom: 6rem">
<div class="entities" style="line-height: 2.5; direction: ltr">
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Sebastian Thrun
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 started working at 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Google Inc.
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 in 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    2007
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
</div>
</figure>
</body>
</html></span>


    
    Using the 'ent' visualizer
    Serving on http://0.0.0.0:5000 ...
    
    Shutting down server on port 5000.
    

#### Training the Named Entity Recognizer Component to  Recognize a new iphones as Entity
[Next we will train the Named Entity Recognizer Component of the pipeline for a new entity called "Gadget"](https://course.spacy.io/en/chapter4)



Content of the file "iphone.json" at https://raw.githubusercontent.com/explosion/spacy-course/master/exercises/en/iphone.json

<br>
[<br>
  "How to preorder the iPhone X",<br>
  "iPhone X is coming",<br>
  "Should I pay $1,000 for the iPhone X?",<br>
  "The iPhone 8 reviews are here",<br>
  "iPhone 11 vs iPhone 8: What's the difference?",<br>
  "I need a new phone! Any tips?"<br>
]<br>



```python
#Source : https://course.spacy.io/en/chapter4
# https://github.com/explosion/spacy-course/tree/master/exercises/en

#Import required libraries
import json
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.tokens import DocBin

# You can access the files at https://raw.githubusercontent.com/explosion/spacy-course/master/exercises/en/iphone.json
#open and read the file contents in json format
with open("iphone.json", encoding="utf8") as f:
    TEXTS = json.loads(f.read())

# use blank model, it has only Tokenizer, no other component
nlp = spacy.blank("en")

# create a Matcher object
matcher = Matcher(nlp.vocab)

# Create two patters that catch the word sequence "iphone x" and "iphone" followed by any digit

# Two tokens whose lowercase forms match "iphone" and "x"
pattern1 = [{"LOWER": "iphone"}, {"LOWER": "x"}]

# Token whose lowercase form matches "iphone" and a digit
pattern2 = [{"LOWER": "iphone"}, {"IS_DIGIT": True}]

# Add patterns to the matcher and create docs with matched entities
matcher.add("GADGET", [pattern1, pattern2])

## initialise list of docs
docs = []

#apply npl to TEXTS read from json file
# apply matcher to identify sequences that satisfy the patterns
for doc in nlp.pipe(TEXTS):
    matches = matcher(doc)
    #create spans of matches in the doc
    spans = [Span(doc, start, end, label=match_id) for match_id, start, end in matches]
    print(spans)
    #store entities
    doc.ents = spans
    docs.append(doc)
```

    [iPhone X]
    [iPhone X]
    [iPhone X]
    [iPhone 8]
    [iPhone 11, iPhone 8]
    []
    


```python
import random
random.shuffle(docs)
train_docs = docs[:len(docs) // 2]
dev_docs = docs[len(docs) // 2:]
```


```python
#Instantiate the DocBin with the list of docs.
#Save the DocBin to a file called train.spacy.
# Create and save a collection of training docs
train_docbin = DocBin(docs=train_docs)
train_docbin.to_disk("C:/spacyfiles/train.spacy")
# Create and save a collection of evaluation docs
dev_docbin = DocBin(docs=dev_docs)
dev_docbin.to_disk("C:/spacyfiles/dev.spacy")

```

<div class="alert alert-block alert-info">
After this we need to generate a config.cfg file with spacy init config command. <br>
For this, use this line of code at your Anaconda prompt. <br>
Issue the command in the same environment where you have installed spacy. <br>
    
<code>python -m spacy init config ./config.cfg --lang en --pipeline ner </code><br>
    
init config: the command to run<br>
config.cfg: output path for the generated config<br>
--lang: language class of the pipeline, e.g. en for English<br>
--pipeline: comma-separated names of components to include<br>
</div>

<div class="alert alert-block alert-info">
Next we train the model by running following command at Anaconda prompt in the environment in which you have installed spacy
<br>

<code>python -m spacy train config.cfg --output C:/spacymodeloutput  --paths.train  C:/spacyfiles/train.spacy  --paths.dev C:/spacyfiles/dev.spacy </code><br>

      
train: the command to run<br>
config.cfg: the path to the config file<br>
--output: the path to the output directory to save the trained pipeline<br>
--paths.train: override with path to the training data, provide the path of train.spacy<br>
--paths.dev: override with path to the evaluation data, provide the path of dev.spacy<br>
    
</div>

## During training

<img src = "https://raw.githubusercontent.com/NimritaKoul/IntroductiontoNLPwith_spaCy/main/TrainingNERwithSpacy.jpg">


### Loading a trained pipeline
output after training is a regular loadable spaCy pipeline
model-last: last trained pipeline
model-best: best trained pipeline
load it with spacy.load


```python
import spacy

nlp = spacy.load("C:/spacymodeloutput/model-best")
doc = nlp("iPhone 11 is better than iPhone 8")
print(doc.ents)
```

    (iPhone 11, iPhone 8)
    

### Packaging your pipeline
spacy package: creates an installable Python package containing your pipeline, easy to version and deploy

<div class="alert alert-block alert-info">
To create a distributable package of your newly trained model use this command at the Ananconda prompt in the same environment in which you installed spacy.

<code>python -m spacy package C:/spacymodeloutput/model-best C:/myspacypackages --name my_pipeline --version 1.0.0</code><br>
Here :<br>
    spacy package: is the command that creates a package<br>
    c:/spacymodeloutput/model-best : is the path of the folder which contains best model generated in training process<br>
    c:/myspacypacakges : is the path in which I want the newly created package to be stored<br>
    my_pipeline : is the name I wish to be given to the created pipeline<br>
        
</div>

### Installing your own pipeline into your Jupyter Notebook
<div class="alert alert-block alert-info">

At the Anaconda Prompt (in your spacy environment) first change the directory to one where your pacakage is stored.<br>

<code>cd C:/myspacypackages/en_my_pipeline-1.0.0</code><br>

Then at the same prompt, use pip install to install the pipeline from tar file.<br>

<code>pip install dist/en_my_pipeline-1.0.0.tar.gz</code><br>
    
</div>


```python
### Load and use your own pipeline
my_nlp_pipeline = spacy.load("en_my_pipeline")
doc = my_nlp_pipeline("iPhone 11 is better than iPhone 8:")
print(doc.ents)
```

    (iPhone 11, iPhone 8)
    


```python

```
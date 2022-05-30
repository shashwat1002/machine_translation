# The Language Model

flags: 

`-train` followed by two subarguments:
    - root directory
    - corpus file name

Used to train model 

`-vocab`
To load or save vocabulary
subargument: vocabulary file

`-model`
To load or save model 
subarg: model file path 

`-test`
to tell it to test
subarg: 
    - toor directory
    - corpus name


# The Machine translation

flags:

`-train`
must be followed by the file paths for both language corpora

`-encoder`

is to refer to load or save encoder model 
must be followed by file path

`-decoder`

is to refer to load or save decoder model 
must be followed by file path  

`-vocab`

to load or save both vocabularies 

`-test`

to test the model and requires both languages testing corpora

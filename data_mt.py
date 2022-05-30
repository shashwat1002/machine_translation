import torch
from torch.utils.data import Dataset
from nltk.tokenize import wordpunct_tokenize, word_tokenize
import json
from icecream import ic

DEVICE = "cuda:0"

class Lang:
    # class to represent either the input or the output language

    def __init__(self, name, device):
        self.name = name
        self.vocab_dict = {}
        self.frequency_dict = {}
        self.vocab_list = []
        self.vocab_size = 0
        self.embedding_list = []
        self.device = device
        self.length_longest_sentence = 0


    def load_from_embedding_file(self, embedding_file_obj, embedding_size):
        full_content = embedding_file_obj.read().strip().split('\n')

        for i in range(len(full_content)):
            i_word = full_content[i].split(' ')[0]
            i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]

            self.vocab_list.append(i_word)
            self.vocab_dict.update({i_word: i})
            self.embedding_list.append(i_embeddings)


        self.vocab_list.append("<sos>")
        # for sos
        self.vocab_list.append("<eos>")
        # eos
        self.vocab_list.append("<unk>")
        # unk

        self.vocab_size = len(self.vocab_list)

        #self.vocab_list = torch.tensor(self.vocab_list).to(self.device)
        self.embedding_list = torch.tensor(self.embedding_list).to(self.device)
        print(self.embedding_list.size())
        sos_emb = torch.zeros(embedding_size).to(self.device)
        eos_emb = torch.zeros(embedding_size).to(self.device)

        unk_emb = torch.mean(self.embedding_list, axis=0, keepdims=True)

        self.embedding_list = torch.cat((self.embedding_list, sos_emb.unsqueeze(0), eos_emb.unsqueeze(0), unk_emb), 0)

    def add_token(self, token):

        try:
            self.frequency_dict[token] += 1
        except KeyError:
            self.vocab_list.append(token)
            self.vocab_dict.update({token: len(self.vocab_list)-1})
            self.frequency_dict.update({token: 1})
            self.vocab_size += 1

    def sort_list_by_frequency(self):
        self.vocab_list = sorted(self.vocab_list, key=lambda token: self.frequency_dict[token], reverse=True)

    def shorten_vocab_list_freq_cutoff(self, frequency_cutoff):

        cutoff_index = 0
        for index in range(self.vocab_size):
            token = self.vocab_list[index]
            freq = self.frequency_dict[token]
            cutoff_index = index
            if freq < frequency_cutoff:
                break
        self.vocab_list = self.vocab_list[:cutoff_index]
        self.vocab_size = len(self.vocab_list)

        self.vocab_dict = {}

        for index, token in enumerate(self.vocab_list):
            self.vocab_dict.update({token: index})

        self.vocab_list.append("<pad>")
        self.vocab_list.append("<sos>")
        self.vocab_list.append("<eos>")
        self.vocab_list.append("<unk>")

        self.vocab_size += 4




    def load_from_corpus(self, corpus_file_obj, frequency_cutoff):

        for line in corpus_file_obj:
            sentence = line.rstrip().lower()
            sentence_tokenized = wordpunct_tokenize(sentence)
            if len(sentence_tokenized) > self.length_longest_sentence:
                self.length_longest_sentence = len(sentence_tokenized)
            for token in sentence_tokenized:
                self.add_token(token)

        self.sort_list_by_frequency()
        self.shorten_vocab_list_freq_cutoff(frequency_cutoff)



    def export_vocabulary(self, vocabulary_file_obj):
        for word in self.frequency_dict:

            word_dict = {"word": word, "freq": self.frequency_dict[word]}
            word_dict_string_rep = json.dumps(word_dict)
            vocabulary_file_obj.write(word_dict_string_rep)
            vocabulary_file_obj.write("\n")

    def import_vocabulary(self, vocab_file_obj, frequency_cutoff):
        word_index = 0

        for line in vocab_file_obj:
            python_rep = json.loads(line)
            word = python_rep["word"]
            frequency = python_rep["freq"]

            self.vocab_dict.update({word: len(self.vocab_list)})
            self.frequency_dict.update({word: frequency})
            self.vocab_list.append(word)
            self.vocab_size += 1

        self.sort_list_by_frequency()
        self.shorten_vocab_list_freq_cutoff(frequency_cutoff)



class MTDataset(Dataset):

    def __init__(self, file1_path, file2_path, lang1_obj, lang2_obj, device, frequency_cutoff, transform=None):

        with open(file1_path, "r") as file1:
            with open(file2_path, "r") as file2:
                self.lang1_lines = file1.readlines()
                self.lang2_lines = file2.readlines()
        self.num_lines = len(self.lang1_lines)
        self.lang1_obj = lang1_obj
        self.lang2_obj = lang2_obj
        self.device = device
        self.frequency_cutoff = frequency_cutoff
    def __len__(self):
        return self.num_lines

    def indexes_from_sentences(self, lang_obj, sentence, frequency_cutoff):
        sentence_tokenized = wordpunct_tokenize(sentence.lower())
        index_list = []

        for token in sentence_tokenized:
            try:
                freq = lang_obj.frequency_dict[token]
                if freq >= frequency_cutoff:
                    index_list.append(lang_obj.vocab_dict[token])
                else:
                    index_list.append(lang_obj.vocab_size-1)
            except KeyError:
                index_list.append(lang_obj.vocab_size-1)
                # <unk>
        return index_list

    def tensor_from_sentence(self, lang_obj, sentence, device, reverse=False):
        sentence_index_list = self.indexes_from_sentences(lang_obj, sentence, self.frequency_cutoff)
        sentence_index_list.append(lang_obj.vocab_size-2)
        sentence_index_list.insert(0, lang_obj.vocab_size-3)
        if reverse:
            sentence_index_list.reverse()
            # this is because seq to seq translation works better on flipping

        tensor_of_indexes = torch.tensor(sentence_index_list, device=self.device)
        return tensor_of_indexes

    def __getitem__(self, index):


        length_longest = max(self.lang1_obj.length_longest_sentence,
                            self.lang2_obj.length_longest_sentence)


        tensor1 = self.tensor_from_sentence(
                                    self.lang1_obj,
                                    self.lang1_lines[index],
                                    self.device,
                                    reverse=False)
        #ic(tensor1.shape)
        to_pad1 = length_longest - len(tensor1)
        #ic(to_pad1)

        tensor2 = self.tensor_from_sentence(
                                    self.lang2_obj,
                                    self.lang2_lines[index],
                                    self.device)

        to_pad2 = length_longest - len(tensor2)
        to_pad = max(to_pad1, to_pad2)
        tensor1_padded = torch.nn.functional.pad(tensor1,
                                                pad=(to_pad1, 0),
                                                mode='constant',
                                                value=self.lang1_obj.vocab_size-4)
        # ic(tensor1_padded.shape)
        # left padding because reversed

        tensor2_padded = torch.nn.functional.pad(tensor2,
                                                pad=(0, to_pad2),
                                                mode='constant',
                                                value=self.lang2_obj.vocab_size-4)

        return (tensor1_padded, tensor2_padded)


# lang1_train_file_path = "./data/ted-talks-corpus/train.en"
# lang2_train_file_path = "./data/ted-talks-corpus/train.fr"
# vocab1_file_path = "./english_vocab.json"
# vocab2_file_path = "./french_vocab.json"

# lang1_obj = Lang("english", DEVICE)
# lang2_obj = Lang("french", DEVICE)
# # with open(lang1_embedding, "r") as lang1_embedding_file_obj:
# #     lang1_obj.load_from_embedding_file(lang1_embedding_file_obj, EMBEDDING_SIZE)
# with open(lang1_train_file_path, "r") as lang1_train_file_obj:
#     lang1_obj.load_from_corpus(lang1_train_file_obj, 2)

# with open(lang2_train_file_path, "r") as lang2_train_file_obj:
#     lang2_obj.load_from_corpus(lang2_train_file_obj, 2)

# with open(vocab1_file_path, "w") as vocab1_file_obj:
#     lang1_obj.export_vocabulary(vocab1_file_obj)

# with open(vocab2_file_path, "w") as vocab2_file_obj:
#     lang2_obj.export_vocabulary(vocab2_file_obj)

# dataset_obj = MTDataset(lang1_train_file_path,
#                     lang2_train_file_path,
#                     lang1_obj,
#                     lang2_obj,
#                     DEVICE,
#                     2,)

# dataloader = torch.utils.data.DataLoader(dataset_obj, batch_size=69)

# print(next(iter(dataloader)))
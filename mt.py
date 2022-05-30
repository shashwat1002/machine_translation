import torch
from torch import nn
from data_mt import *
from torch.utils.data import DataLoader
import sys
from icecream import ic
import random
from torchtext.data.metrics import bleu_score

EMBEDDING_SIZE = 100
DEVICE = "cuda:0"

MAX_BATCH_SIZE = 45

NUM_EPOCHS = 50

FREQUENCY_CUTOFF = 2

NUM_LSTM_LAYERS = 1

TEACHER_FORCED = 0.4

class Encoder(nn.Module):

    def __init__(self, hidden_size, lang):

        # hidden_size = EMBEDDING_SIZE

        super().__init__()

        self.vocab_size = lang.vocab_size
        self.hidden_size = hidden_size

        # self.embedding = nn.Embedding(
        #                     self.vocab_size,
        #                     EMBEDDING_SIZE)

        #self.embedding = nn.Embedding.from_pretrained(lang.embedding_list.float())
        self.embedding = nn.Embedding(self.vocab_size, EMBEDDING_SIZE)
        self.lstm = nn.LSTM(
                    input_size=EMBEDDING_SIZE,
                    hidden_size=self.hidden_size,
                    num_layers=NUM_LSTM_LAYERS,
                    batch_first=True
        )
        self.lang = lang

    def forward(self, x, prev_state=None):
        # ic(x.shape)
        embedded = self.embedding(x)
        # ic(embedded.shape)

        output, state = self.lstm(embedded)
        # ic(output.shape, state[0].shape)
        return output, state

    def init_hidden(self):
        return (torch.zeros(NUM_LSTM_LAYERS, 1, self.hidden_size, device=DEVICE),
                torch.zeros(NUM_LSTM_LAYERS, 1, self.hidden_size, device=DEVICE))


class Decoder(nn.Module):
    def __init__(self, hidden_size, lang):
        super().__init__()

        self.hidden_size = hidden_size
        output_size = lang.vocab_size

        self.embedding = nn.Embedding(
            output_size, self.hidden_size
        )
        # self.relu = nn.ReLU()
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)

        self.final_layer = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.lang = lang


    def forward(self, x, prev_state):
        #ic("decoder:",  x.shape)
        embedded = self.embedding(x)
        #ic("decoder embedding", embedded.shape)
        output, state = self.lstm(embedded, prev_state)
        #ic(output.shape)

        logits = self.final_layer(output)
        #ic("decoder logits", logits.shape)
        # prob_disr = self.softmax(logits)

        return logits, state

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)

def train_step(input_tensor,
                target_tensor,
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                loss_fun,
                device,
                max_batch_size=MAX_BATCH_SIZE,):
    # ic(input_tensor.shape, target_tensor.shape)

    encoder_state = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # reset both optimizers

    encoder_outputs = torch.zeros(MAX_BATCH_SIZE, encoder.hidden_size, device=DEVICE)

    input_length = input_tensor.size(-1)
    target_length = target_tensor.size(-1)
    # for i in range(input_length):
    #     val = input_tensor[i].detach()
    #     encoder_output, encoder_state = encoder(
    #         val, encoder_state
    #     )
    #     #encoder_outputs[i] = encouter_output[0, 0]
    encoder_output, (hidden_state, cell_state) = encoder(input_tensor)


    # ic(hidden_state.shape)
    decoder_input = torch.tensor([decoder.lang.vocab_size-3]*target_tensor.shape[0], device=DEVICE)
    decoder_input = decoder_input.view(-1, 1)

    decoder_state = (hidden_state, cell_state)
    # the last state of the encoder is the IR representation
    loss = torch.tensor(0.0, device=DEVICE)

    for i in range(1, target_length):
        # ic(decoder_input.shape, decoder_state[0].shape)
        decoder_output, decoder_state = decoder(
            decoder_input,
            decoder_state
        )
        topv, topi = decoder_output.topk(1)
        # get the highest probability word

        if i == 1:
            ran = random.random()
            if ran <= TEACHER_FORCED:
                decoder_input = target_tensor[:, 1]

            else:
                decoder_input = topi.squeeze().detach()
        else:
            decoder_input = topi.squeeze().detach()
        decoder_input = decoder_input.view(-1, 1)
        # ic("end of loop", decoder_input.shape)
        # index of the most probably word will be sent now
        # ic(decoder_output.view((-1, decoder_output.size(-1))).shape)
        decoder_output_mid_squeezed = decoder_output.view((-1, decoder_output.size(-1)))
        # ic(i, target_tensor[:, i])

        loss_temp = loss_fun(decoder_output_mid_squeezed, target_tensor[:, i])
        if not torch.isnan(loss_temp):
            loss += loss_temp

    loss.backward()

    ic(loss)


    encoder_optimizer.step()
    decoder_optimizer.step()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    return loss.item()

def train_epoch(data_loader_obj,
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                loss_fun,
                device):

    loss = 0
    for (input_tensor, target_tensor) in data_loader_obj:
        loss += train_step(input_tensor,
                        target_tensor,
                        encoder,
                        decoder,
                        encoder_optimizer,
                        decoder_optimizer,
                        loss_fun,
                        DEVICE)
    avg_loss = loss / len(data_loader_obj)
    print(avg_loss)

def train_model(encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                loss_fun,
                device,
                encoder_path,
                decoder_path,
                dataset,
                num_epochs=NUM_EPOCHS):


    for epoch in range(num_epochs):
        print(f"{epoch+1}")
        data_loader_obj = DataLoader(dataset, shuffle=True, batch_size=MAX_BATCH_SIZE)
        train_epoch(data_loader_obj, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fun, DEVICE)
        print("------------")

        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)

def translate(sentence, encoder, decoder):

    #sentence_tokenized = wordpunct_tokenize(sentence.rstrip().lower())
    index_list = MTDataset.indexes_from_sentences(None, encoder.lang, sentence.rstrip(), FREQUENCY_CUTOFF)
    index_list.reverse()
    index_list = torch.tensor(index_list, device=DEVICE).unsqueeze(dim=0)
    print(index_list.shape)
    encoder_state = encoder.init_hidden()
    encoder_output = None

    encoder_output, encoder_state = encoder(index_list, encoder_state)

    decoder_input = torch.tensor([[decoder.lang.vocab_size-3]], device=DEVICE)

    decoder_state = encoder_state
    # the last state of the encoder is the IR representation
    final_out = []
    counter = 0
    while decoder_input != decoder.lang.vocab_size-2 and counter < 15:

        decoder_output, decoder_state = decoder(
            decoder_input,
            decoder_state
        )
        topv, topi = decoder_output.topk(1)
        # get the highest probability word

        decoder_input = topi.squeeze().detach().view((-1, 1))
        print(decoder_input.item())
        if decoder_input.item() != decoder.lang.vocab_size-4:
            final_out.append(decoder_input.item())
        counter += 1

    final_string_out = []

    for index in final_out:
        word = decoder.lang.vocab_list[index]
        final_string_out.append(word)

    return final_string_out




def main():

    train = False
    test = False
    demo = False

    lang1_train_file_path = None
    lang1_train_file_obj = None

    lang2_train_file_path = None
    lang2_train_file_obj = None


    test_file_path = None
    test_file_obj = None

    encoder_file_path = None
    decoder_file_path = None

    lang1_embedding = False
    lang2_embedding = False

    argument_index = 0

    vocab1_file_path = None
    vocab2_file_path = None

    bleu_file_path = None

    while argument_index < len(sys.argv):

        if sys.argv[argument_index] == "-train":
            train = True
            argument_index += 1
            lang1_train_file_path = sys.argv[argument_index]
            argument_index += 1
            lang2_train_file_path = sys.argv[argument_index]

        if sys.argv[argument_index] == "-encoder":
            argument_index += 1
            encoder_file_path = sys.argv[argument_index]

        if sys.argv[argument_index] == "-decoder":
            argument_index += 1
            decoder_file_path = sys.argv[argument_index]

        if sys.argv[argument_index] == "-embedding1":
            argument_index += 1
            lang1_embedding = sys.argv[argument_index]

        if sys.argv[argument_index] == "-vocab":
            argument_index += 1
            vocab1_file_path = sys.argv[argument_index]
            argument_index += 1
            vocab2_file_path = sys.argv[argument_index]
        if sys.argv[argument_index] == "-demo":
            #argument_index += 1
            demo = True

        if sys.argv[argument_index] == "-test":
            test = True
            argument_index += 1
            lang1_train_file_path = sys.argv[argument_index]
            argument_index += 1
            lang2_train_file_path = sys.argv[argument_index]
            argument_index += 1
            bleu_file_path = sys.argv[argument_index]


        # if sys.argv[argument_index] == "-corpus1":
        #     argument_index += 1
        #     lang1_train_file_path = sys.argv[argument_index]

        # if sys.argv[argument_index] == "-corpus2":
        #     argument_index += 1
        #     lang2_train_file_path = sys.argv[argument_index]


        argument_index += 1


    if train:

        lang1_obj = Lang("english", DEVICE)
        lang2_obj = Lang("french", DEVICE)
        # with open(lang1_embedding, "r") as lang1_embedding_file_obj:
        #     lang1_obj.load_from_embedding_file(lang1_embedding_file_obj, EMBEDDING_SIZE)
        with open(lang1_train_file_path, "r") as lang1_train_file_obj:
            lang1_obj.load_from_corpus(lang1_train_file_obj, FREQUENCY_CUTOFF)

        with open(lang2_train_file_path, "r") as lang2_train_file_obj:
            lang2_obj.load_from_corpus(lang2_train_file_obj, FREQUENCY_CUTOFF)

        with open(vocab1_file_path, "w") as vocab1_file_obj:
            lang1_obj.export_vocabulary(vocab1_file_obj)

        with open(vocab2_file_path, "w") as vocab2_file_obj:
            lang2_obj.export_vocabulary(vocab2_file_obj)

        dataset_obj = MTDataset(lang1_train_file_path,
                            lang2_train_file_path,
                            lang1_obj,
                            lang2_obj,
                            DEVICE,
                            FREQUENCY_CUTOFF,)
        encoder, decoder = None, None

        encoder = Encoder(EMBEDDING_SIZE, lang1_obj).to(DEVICE)
        decoder = Decoder(EMBEDDING_SIZE, lang2_obj).to(DEVICE)

        try:
            encoder.load_state_dict(torch.load(encoder_file_path))
            decoder.load_state_dict(torch.load(decoder_file_path))
        except FileNotFoundError:
            pass

        encoder_optimizer = torch.optim.Adam(params=encoder.parameters(), lr=1e-6)
        decoder_optimizer = torch.optim.Adam(params=decoder.parameters(), lr=1e-6)

        loss_fun = nn.CrossEntropyLoss(ignore_index=lang2_obj.vocab_size-4)
        train_model(encoder,
                    decoder,
                    encoder_optimizer,
                    decoder_optimizer,
                    loss_fun,
                    DEVICE,
                    encoder_file_path,
                    decoder_file_path,
                    dataset_obj)

    elif demo:
        lang1_obj = Lang("english", DEVICE)
        lang2_obj = Lang("french", DEVICE)

        with open(vocab1_file_path, "r") as vocab1_file_obj:
            lang1_obj.import_vocabulary(vocab1_file_obj, FREQUENCY_CUTOFF)

        with open(vocab2_file_path, "r") as vocab2_file_obj:
            lang2_obj.import_vocabulary(vocab2_file_obj, FREQUENCY_CUTOFF)

        encoder = Encoder(EMBEDDING_SIZE, lang1_obj).to(DEVICE)
        encoder.load_state_dict(torch.load(encoder_file_path))
        encoder.eval()

        decoder = Decoder(EMBEDDING_SIZE, lang2_obj).to(DEVICE)
        decoder.load_state_dict(torch.load(decoder_file_path))
        decoder.eval()

        sentence = input()
        translated = translate(sentence, encoder, decoder)

        print(translated)

    else:
        lang1_obj = Lang("english", DEVICE)
        lang2_obj = Lang("french", DEVICE)

        with open(vocab1_file_path, "r") as vocab1_file_obj:
            lang1_obj.import_vocabulary(vocab1_file_obj, FREQUENCY_CUTOFF)

        with open(vocab2_file_path, "r") as vocab2_file_obj:
            lang2_obj.import_vocabulary(vocab2_file_obj, FREQUENCY_CUTOFF)

        encoder = Encoder(EMBEDDING_SIZE, lang1_obj).to(DEVICE)
        encoder.load_state_dict(torch.load(encoder_file_path))
        encoder.eval()

        decoder = Decoder(EMBEDDING_SIZE, lang2_obj).to(DEVICE)
        decoder.load_state_dict(torch.load(decoder_file_path))
        decoder.eval()

        translated = []
        with open(lang1_train_file_path, "r") as corpus1_file_obj:
            for line in corpus1_file_obj:
                print(line)
                translated_sentence = translate(line, encoder, decoder)
                translated.append(translated_sentence)

        comparison_standard = []
        with open(lang2_train_file_path, "r") as corpus2_file_obj:
            for line in corpus2_file_obj:
                sentence_tokenized = wordpunct_tokenize(line.rstrip().lower())
                sentence_tokenized.append("<eos>")
                comparison_standard.append([sentence_tokenized,])
        print(comparison_standard)
        print(translated)
        assert(translated is not None)
        assert(comparison_standard is not None)
        score = bleu_score(candidate_corpus=translated, references_corpus=comparison_standard)
        print(score)

        with open(bleu_file_path, "w") as bleu_file_object:
            for i in range(len(comparison_standard)):
                sentence_main = comparison_standard[i]
                sentence_translated = translated[i]
                score = bleu_score([sentence_translated], [sentence_main])
                bleu_file_object.write(f"{sentence_main} \t {sentence_translated} {score}\n")





main()
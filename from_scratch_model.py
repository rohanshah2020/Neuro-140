{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "metadata": {
        "id": "J8DF8S_mxPBo"
      },
      "cell_type": "code",
      "source": [
        "import itertools, os, time , datetime\n",
        "import numpy as np\n",
        "import spacy\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "from torchtext import data, datasets\n",
        "from torchtext.vocab import Vectors, GloVe\n",
        "use_gpu = torch.cuda.is_available()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "id": "2iFfBRzL4vl9"
      },
      "cell_type": "code",
      "source": [
        "def load_embeddings(SRC, TGT, np_src_file, tgt_file):\n",
        "    emb_tr_src = torch.from_numpy(np.load(np_src_file))\n",
        "    emb_tr_tgt = torch.from_numpy(np.load(np_tgt_file))\n",
        "    return emb_tr_src, emb_tr_tgt"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "id": "VMKaLXm44sav"
      },
      "cell_type": "code",
      "source": [
        "class EncoderLSTM(nn.Module):\n",
        "    def __init__(self, embedding, h_dim, num_layers, dropout_p=0.0, bidirectional=True):\n",
        "        super(EncoderLSTM, self).__init__()\n",
        "        self.vocab_size, self.embedding_size = embedding.size()\n",
        "        self.num_layers, self.h_dim, self.dropout_p, self.bidirectional = num_layers, h_dim, dropout_p, bidirectional \n",
        "\n",
        "        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)\n",
        "        self.embedding.weight.data.copy_(embedding)\n",
        "        self.lstm = nn.LSTM(self.embedding_size, self.h_dim, self.num_layers, dropout=self.dropout_p, bidirectional=bidirectional)\n",
        "        self.dropout = nn.Dropout(dropout_p)\n",
        "\n",
        "    def forward(self, x):\n",
        "        x = self.dropout(self.embedding(x)) \n",
        "        h0 = self.init_hidden(x.size(1))\n",
        "        memory_bank, h = self.lstm(x, h0) \n",
        "        return memory_bank, h\n",
        "\n",
        "    def init_hidden(self, batch_size):\n",
        "        num_layers = self.num_layers * 2 if self.bidirectional else self.num_layers\n",
        "        init = torch.zeros(num_layers, batch_size, self.h_dim)\n",
        "        init = init.cuda() if use_gpu else init\n",
        "        h0 = (init, init.clone())"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "id": "NuH-ZC616ThJ"
      },
      "cell_type": "code",
      "source": [
        "class Attention(nn.Module):\n",
        "    def __init__(self, pad_token=1, bidirectional=True, h_dim=300):\n",
        "        super(Attention, self).__init__()\n",
        "        self.bidirectional, self.h_dim, self.pad_token = bidirectional, h_dim, pad_token\n",
        "        self.softmax = nn.Softmax(dim=1)\n",
        "\n",
        "    def forward(self, in_e, out_e, out_d):\n",
        "\n",
        "        if self.bidirectional:\n",
        "            out_e = out_e.contiguous().view(out_e.size(0), out_e.size(1), 2, -1).sum(2).view(out_e.size(0), out_e.size(1), -1)\n",
        "            \n",
        "        out_e = out_e.transpose(0,1) \n",
        "        out_d = out_d.transpose(0,1) \n",
        "\n",
        "        attn = out_e.bmm(out_d.transpose(1,2)) \n",
        "        attn = self.softmax(attn).transpose(1,2) \n",
        "\n",
        "        context = attn.bmm(out_e) \n",
        "        context = context.transpose(0,1) \n",
        "        return context"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "id": "W8zFya8t6UwP"
      },
      "cell_type": "code",
      "source": [
        "class DecoderLSTM(nn.Module):\n",
        "    def __init__(self, embedding, h_dim, num_layers, dropout_p=0.0):\n",
        "        super(DecoderLSTM, self).__init__()\n",
        "        self.vocab_size, self.embedding_size = embedding.size()\n",
        "        self.num_layers, self.h_dim, self.dropout_p = num_layers, h_dim, dropout_p\n",
        "        \n",
        "        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)\n",
        "        self.embedding.weight.data.copy_(embedding) \n",
        "        self.lstm = nn.LSTM(self.embedding_size, self.h_dim, self.num_layers, dropout=self.dropout_p)\n",
        "        self.dropout = nn.Dropout(self.dropout_p)\n",
        "    \n",
        "    def forward(self, x, h0):\n",
        "        x = self.embedding(x)\n",
        "        x = self.dropout(x)\n",
        "        out, h = self.lstm(x, h0)\n",
        "        return out, h"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "id": "y5N93D6XB9M-"
      },
      "cell_type": "code",
      "source": [
        "class Seq2seq(nn.Module):\n",
        "    def __init__(self, embedding_src, embedding_tgt, h_dim, num_layers, dropout_p, bi, tokens_bos_eos_pad_unk=[0,1,2,3]):\n",
        "        super(Seq2seq, self).__init__()\n",
        "    \n",
        "        self.h_dim = h_dim\n",
        "        self.vocab_size_tgt, self.emb_dim_tgt = embedding_tgt.size()\n",
        "        self.bos_token, self.eos_token, self.pad_token, self.unk_token = tokens_bos_eos_pad_unk\n",
        "\n",
        "        self.encoder = EncoderLSTM(embedding_src, h_dim, num_layers, dropout_p=dropout_p, bidirectional=bi)\n",
        "        self.decoder = DecoderLSTM(embedding_tgt, h_dim, num_layers * 2 if bi else num_layers, dropout_p=dropout_p)\n",
        "        self.attention = Attention(pad_token=self.pad_token, bidirectional=bi, h_dim=self.h_dim)\n",
        "\n",
        "        self.linear1 = nn.Linear(2 * self.h_dim, self.emb_dim_tgt)\n",
        "        self.tanh = nn.Tanh()\n",
        "        self.dropout = nn.Dropout(dropout_p)\n",
        "        self.linear2 = nn.Linear(self.emb_dim_tgt, self.vocab_size_tgt)\n",
        "        \n",
        "        if self.decoder.embedding.weight.size() == self.linear2.weight.size():\n",
        "            self.linear2.weight = self.decoder.embedding.weight\n",
        "\n",
        "    def forward(self, src, tgt):\n",
        "        if use_gpu: src = src.cuda()\n",
        "        \n",
        "        out_e, final_e = self.encoder(src)\n",
        "\n",
        "        out_d, final_d = self.decoder(tgt, final_e)\n",
        "        \n",
        "        context = self.attention(src, out_e, out_d)\n",
        "        out_cat = torch.cat((out_d, context), dim=2) \n",
        "        \n",
        "        out = self.linear1(out_cat)\n",
        "        out = self.dropout(self.tanh(out))\n",
        "        out = self.linear2(out)\n",
        "        return out\n",
        "\n",
        "    def predict(self, src, beam_size=1): \n",
        "        beam_outputs = self.beam_search(src, beam_size, max_len=30) # returns top candidates in tuble form\n",
        "        top1 = beam_outputs[1][1] \n",
        "        return top1 # returns a subsection of those candidates\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        " def beam_search(self, src, beam_size, max_len, remove_tokens=[]):\n",
        "        if use_gpu: \n",
        "          src = src.cuda()\n",
        "        outputs_e, states = self.encoder(src) \n",
        "        init_sent = [self.bos_token]\n",
        "        init_lprob = -1e10\n",
        "        best_candidates = [(init_lprob, init_sent, states)] \n",
        "        \n",
        "        k = beam_size \n",
        "        for length in range(max_len):\n",
        "            candidates = [] \n",
        "            for lprob, sentence, current_state in best_candidates:\n",
        "                last_word = sentence[-1]\n",
        "                if last_word != self.eos_token:\n",
        "                    last_word_input = torch.LongTensor([last_word]).view(1,1)\n",
        "                    if use_gpu: last_word_input = last_word_input.cuda()\n",
        "                    outputs_d, new_state = self.decoder(last_word_input, current_state)\n",
        "                    context = self.attention(src, outputs_e, outputs_d)\n",
        "                    out_cat = torch.cat((outputs_d, context), dim=2)\n",
        "                    x = self.linear1(out_cat)\n",
        "                    x = self.dropout(self.tanh(x))\n",
        "                    x = self.linear2(x)\n",
        "                    x = x.squeeze().data.clone()\n",
        "                    for t in remove_tokens: x[t] = -10e5\n",
        "                    lprobs = torch.log(x.exp() / x.exp().sum())\n",
        "                    for index in torch.topk(lprobs, k)[1]: \n",
        "                        candidate = (float(lprobs[index]) + lprob, sentence + [index], new_state) \n",
        "                        candidates.append(option)\n",
        "                else:\n",
        "                    candidates.append((lprob, sentence, current_state))\n",
        "            candidates.sort(key = lambda x: x[0], reverse=True) # sort by lprob\n",
        "            best_candidates = candidates[:k] \n",
        "        best_candidates.sort(key = lambda x: x[0])\n",
        "        return best_candidates\n"
      ],
      "metadata": {
        "id": "gqzShe-ngE8W"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "id": "fc_JGveeBfG_"
      },
      "cell_type": "code",
      "source": [
        "def train(train_iter, val_iter, model, criterion, optimizer, num_epochs):  \n",
        "    for epoch in range(num_epochs):\n",
        "      \n",
        "        with torch.no_grad():\n",
        "          val_loss = validate(val_iter, model, criterion) \n",
        "          print('Validating Epoch [{e}/{num_e}]\\t Average loss: {l:.3f}\\t Perplexity: {p:.3f}'.format(\n",
        "            e=epoch, num_e=num_epochs, l=val_loss, p=torch.FloatTensor([val_loss]).exp().item()))\n",
        "\n",
        "        model.train()\n",
        "        losses = AverageMeter()\n",
        "        for i, batch in enumerate(train_iter): \n",
        "            src = batch.src.cuda() if use_gpu else batch.src\n",
        "            tgt = batch.trg.cuda() if use_gpu else batch.trg\n",
        "            \n",
        "            # Includes backprop and optimizer\n",
        "            model.zero_grad()\n",
        "            scores = model(src, tgt)\n",
        "            scores = scores[:-1]\n",
        "            tgt = tgt[1:]           \n",
        "\n",
        "            scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))\n",
        "            tgt = tgt.view(scores.size(0))\n",
        "            loss = criterion(scores, tgt) \n",
        "            loss.backward()\n",
        "            losses.update(loss.item())\n",
        "            optimizer.step()\n",
        "\n",
        "            if i % 1000 == 10:\n",
        "                print('''Epoch [{e}/{num_e}]\\t Batch [{b}/{num_b}]\\t Loss: {l:.3f}'''.format(e=epoch+1, num_e=num_epochs, b=i, num_b=len(train_iter), l=losses.avg))\n",
        "\n",
        "        print('''Epoch [{e}/{num_e}] complete. Loss: {l:.3f}'''.format(e=epoch+1, num_e=num_epochs, l=losses.avg))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "id": "1s7Dk9564sUr"
      },
      "cell_type": "code",
      "source": [
        "def validate(val_iter, model, criterion):\n",
        "    model.eval()\n",
        "    losses = AverageMeter()\n",
        "    for i, batch in enumerate(val_iter):\n",
        "        src = batch.src.cuda() if use_gpu else batch.src\n",
        "        tgt = batch.trg.cuda() if use_gpu else batch.trg\n",
        "        \n",
        "        scores = model(src, tgt)\n",
        "        scores = scores[:-1]\n",
        "        tgt = tgt[1:]           \n",
        "        \n",
        "        scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))\n",
        "        tgt = tgt.view(scores.size(0))\n",
        "        num_words = (tgt != 0).float().sum()\n",
        "        \n",
        "\n",
        "        loss = criterion(scores, tgt) \n",
        "        losses.update(loss.item())\n",
        "    \n",
        "    return losses.avg"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "id": "uGgeAIPGEMVp"
      },
      "cell_type": "code",
      "source": [
        "def predict_from_text(model, input_sentence, SRC, TGT):\n",
        "    sent_german = input_sentence.split(' ') \n",
        "    sent_indices = [SRC.vocab.stoi[word] if word in SRC.vocab.stoi else SRC.vocab.stoi['<unk>'] for word in sent_german]\n",
        "    sent = torch.LongTensor([sent_indices])\n",
        "    if use_gpu: sent = sent.cuda()\n",
        "    sent = sent.view(-1,1) \n",
        "    print('German: ' + ' '.join([SRC.vocab.itos[index] for index in sent_indices])) \n",
        "    pred = model.predict(sent, beam_size=15) \n",
        "    out = ' '.join([TGT.vocab.itos[index] for index in pred[1:-1]])\n",
        "    print('English: ' + out)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "id": "1PU0wtkPRasL"
      },
      "cell_type": "code",
      "source": [
        "\n",
        "embedding_src, embedding_tgt = load_embeddings(SRC, TGT, 'emb-13353-de.npy', 'emb-11560-en.npy')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "id": "7JacYrR2Fnwt"
      },
      "cell_type": "code",
      "source": [
        "\n",
        "tokens = [TGT.vocab.stoi[x] for x in ['<s>', '</s>', '<pad>', '<unk>']]\n",
        "model = Seq2seq(embedding_src, embedding_tgt, 300, 2, 0.3, True, tokens_bos_eos_pad_unk=tokens)\n",
        "model = model.cuda() if use_gpu else model"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "id": "kT4EGSBNGGUu"
      },
      "cell_type": "code",
      "source": [
        "\n",
        "weight = torch.ones(len(TGT.vocab))\n",
        "weight[TGT.vocab.stoi['<pad>']] = 0\n",
        "weight = weight.cuda() if use_gpu else weight"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "id": "INe-LU06GGJr"
      },
      "cell_type": "code",
      "source": [
        "\n",
        "criterion = nn.CrossEntropyLoss(weight=weight)\n",
        "optimizer = torch.optim.Adam(model.parameters(), lr=2e-3) "
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "id": "f6qdcw09GF-I",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 942
        },
        "outputId": "376ba5de-1581-42f0-9e5b-0b46adafd2a6"
      },
      "cell_type": "code",
      "source": [
        "train(train_iter, val_iter, model, criterion, optimizer, 100)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Validating Epoch [0/50]\t Average loss: 3.439\t Perplexity: 31.152\n",
            "Epoch [1/50]\t Batch [10/7443]\t Loss: 3.980\n",
            "Epoch [1/50]\t Batch [1010/7443]\t Loss: 3.609\n",
            "Epoch [1/50]\t Batch [2010/7443]\t Loss: 3.446\n",
            "Epoch [1/50]\t Batch [3010/7443]\t Loss: 3.336\n",
            "Epoch [1/50]\t Batch [4010/7443]\t Loss: 3.252\n",
            "Epoch [1/50]\t Batch [5010/7443]\t Loss: 3.182\n",
            "Epoch [1/50]\t Batch [6010/7443]\t Loss: 3.122\n",
            "Epoch [1/50]\t Batch [7010/7443]\t Loss: 3.073\n",
            "Epoch [1/50] complete. Loss: 3.052\n",
            "Validating Epoch [1/50]\t Average loss: 2.274\t Perplexity: 9.716\n",
            "Epoch [2/50]\t Batch [10/7443]\t Loss: 2.552\n",
            "Epoch [2/50]\t Batch [1010/7443]\t Loss: 2.550\n",
            "Epoch [2/50]\t Batch [2010/7443]\t Loss: 2.556\n",
            "Epoch [2/50]\t Batch [3010/7443]\t Loss: 2.548\n",
            "Epoch [2/50]\t Batch [4010/7443]\t Loss: 2.542\n",
            "Epoch [2/50]\t Batch [5010/7443]\t Loss: 2.538\n",
            "Epoch [2/50]\t Batch [6010/7443]\t Loss: 2.537\n",
            "Epoch [2/50]\t Batch [7010/7443]\t Loss: 2.533\n",
            "Epoch [2/50] complete. Loss: 2.531\n",
            "Validating Epoch [2/50]\t Average loss: 2.163\t Perplexity: 8.701\n",
            "Epoch [3/50]\t Batch [10/7443]\t Loss: 2.337\n",
            "Epoch [3/50]\t Batch [1010/7443]\t Loss: 2.319\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "error",
          "ename": "KeyboardInterrupt",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-61-0c793999c16c>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mtrain\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrain_iter\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mval_iter\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcriterion\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0moptimizer\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m50\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
            "\u001b[0;32m<ipython-input-59-9a5864a40aa3>\u001b[0m in \u001b[0;36mtrain\u001b[0;34m(train_iter, val_iter, model, criterion, optimizer, num_epochs)\u001b[0m\n\u001b[1;32m     29\u001b[0m             \u001b[0;31m# Pass through loss function\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     30\u001b[0m             \u001b[0mloss\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcriterion\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mscores\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtgt\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 31\u001b[0;31m             \u001b[0mloss\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbackward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     32\u001b[0m             \u001b[0mlosses\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mloss\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mitem\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     33\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/torch/tensor.py\u001b[0m in \u001b[0;36mbackward\u001b[0;34m(self, gradient, retain_graph, create_graph)\u001b[0m\n\u001b[1;32m     91\u001b[0m                 \u001b[0mproducts\u001b[0m\u001b[0;34m.\u001b[0m \u001b[0mDefaults\u001b[0m \u001b[0mto\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m`\u001b[0m\u001b[0;31m`\u001b[0m\u001b[0;32mFalse\u001b[0m\u001b[0;31m`\u001b[0m\u001b[0;31m`\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     92\u001b[0m         \"\"\"\n\u001b[0;32m---> 93\u001b[0;31m         \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mautograd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbackward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgradient\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mretain_graph\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcreate_graph\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     94\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     95\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mregister_hook\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mhook\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/torch/autograd/__init__.py\u001b[0m in \u001b[0;36mbackward\u001b[0;34m(tensors, grad_tensors, retain_graph, create_graph, grad_variables)\u001b[0m\n\u001b[1;32m     87\u001b[0m     Variable._execution_engine.run_backward(\n\u001b[1;32m     88\u001b[0m         \u001b[0mtensors\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgrad_tensors\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mretain_graph\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcreate_graph\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 89\u001b[0;31m         allow_unreachable=True)  # allow_unreachable flag\n\u001b[0m\u001b[1;32m     90\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     91\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
          ]
        }
      ]
    },
    {
      "metadata": {
        "id": "vVmld246XxBb"
      },
      "cell_type": "code",
      "source": [
        "model.load_state_dict(torch.load('model.pkl'))\n",
        "model = model.cuda() if use_gpu else model"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "id": "ksarzeeFg8xG",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        },
        "outputId": "f621a159-2377-4486-9eac-2db4edc8f726"
      },
      "cell_type": "code",
      "source": [
        "with torch.no_grad():\n",
        "  val_loss = validate(val_iter, model, criterion) \n",
        "  print('Average loss: {l:.3f}\\t Perplexity: {p:.3f}'.format(l=val_loss, p=torch.FloatTensor([val_loss]).exp().item()))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Average loss: 1.865\t Perplexity: 6.459\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "metadata": {
        "id": "84sqEuVlhQij",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 52
        },
        "outputId": "f9fe97d3-7c6e-4f1f-cb04-8fde0d3810d6"
      },
      "cell_type": "code",
      "source": [
        "input = \"Ich kenne nur Berge, ich bleibe in den Bergen und ich liebe die Berge .\"\n",
        "predict_from_text(model, input, SRC, TGT)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "German: Ich kenne nur <unk> ich bleibe in den Bergen und ich liebe die Berge .\n",
            "English: I only know I 'm staying in the hills , and I love the mountains .\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}
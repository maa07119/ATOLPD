import torch
import string
from datasets import load_dataset
from models.gpt_model import GPTLanguageModel
from dataset.text_dataset import TextDataset
from torcheval.metrics.text import Perplexity
from tqdm import tqdm

device = "cuda:0"


# Load the WikiText-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-v1")

train_text = dataset['train']['text']
val_text = dataset['validation']['text']
test_text = dataset['test']['text']

# Combine the text into a single string
train_text = ' '.join(train_text)
val_text = ' '.join(val_text)
test_text = ' '.join(test_text)

train_text = train_text.replace("<unk>", "")
val_text = train_text.replace("<unk>", "")
test_text = train_text.replace("<unk>", "")

def get_vocab(text):
    vocab = set(text)
    return vocab

def check_correct_word(word, vocab):
    for c in word:
        if c not in vocab:
            return False
    return True

def clean_text(text, vocab):
    vocab = set(vocab)
    s = ""
    for word in text.split(" "):
        if check_correct_word(word, vocab):
            s += " " + word
    return s[1:]



vocab = string.ascii_lowercase + string.ascii_uppercase + " " + "\n" + string.punctuation + string.digits

vocab_size = len(vocab)

print(train_text[:100])
train_text = clean_text(train_text, vocab)
test_text = clean_text(test_text, vocab)

def train(model, trainloader, testloader, optimizer, epochs=100, s = 4096):
    print("Training started")
    criterion = torch.nn.CrossEntropyLoss()
    samples_processed = 0
    test_loss = 999
    model.train()

    for i, (x, y) in tqdm(enumerate(trainloader)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # source_mask = (x != 0).unsqueeze(1).unsqueeze(2)
        output, _ = model(x.squeeze(1))
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        #TODO:
        #  Add tensorboard
        batch_size = x.size(0)
        samples_processed += batch_size
        # if samples_processed % s == 0:
        if i % 500 == 0:
            print("*" * 100)
            print('Iteration: %d, Loss: %f' % (i, loss.item()))
            acc, t_loss = evaluate(model, testloader)
            # save model here
            if t_loss < test_loss:
                test_loss = t_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'accuracy': acc,
                    'test_loss':test_loss,
                }, f'pretrained_transformer_{i}_{t_loss}.pt')
            model.train()
    
        

            
def evaluate(model, testloader):
    model.eval()
    # criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0
    total_tokens = 0
    flag = False
    correct_characters = 0
    total_characters = 0
    metric=Perplexity()
 
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            output, _ = model(x.squeeze(1))
            metric.update(output.cpu(), y.cpu())
            predicted = torch.argmax(output[:, -1, :], axis=1)
            output = output.view(-1, vocab_size)
            loss = criterion(output, y.view(-1))
            correct_characters += (predicted == y[:, -1]).sum().item()
            total_characters += predicted.shape[0]
            total_loss += loss.item()
            # total_tokens += y.numel()

    perplexity = metric.compute().item()
    print("Perplexity:", perplexity)
    test_loss = total_loss / len(testloader)
    print("Test loss mean:", test_loss)
    # total_tokens = torch.tensor(total_tokens, dtype=torch.float32)
    # perplexity = torch.exp((total_loss / total_tokens))
    # print('Perplexity: %f' % perplexity)
    character_accuracy = correct_characters / total_characters
    print('Character Level Accuracy: %.2f%%' % (character_accuracy * 100))
    return character_accuracy, test_loss


seq_length = 64
batch_size = 128
train_dataset = TextDataset(train_text, vocab, seq_length, is_train = True)
test_dataset = TextDataset(test_text, vocab, seq_length, is_train = False)

max_iterations = 1500
sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=batch_size * max_iterations)

trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler)
testloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

model = GPTLanguageModel(vocab_size, seq_length, n_embd=256, n_head=6, n_layer=6, dropout=0.1)
print(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
# evaluate(model, testloader)
train(model, trainloader, testloader, optimizer)

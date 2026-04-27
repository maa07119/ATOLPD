import torch
import numpy as np
from torch.utils.data import random_split
import torch.nn.functional as F
from models.gpt_model import GPTLanguageModel
from dataset.text_dataset import get_dataset
from torcheval.metrics.text import Perplexity
from torch.utils.tensorboard import SummaryWriter
import random
from argparse import ArgumentParser
from tqdm import tqdm


# setting random seeds
seed = 12345
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train(model, trainloader, testloader, optimizer, epochs=100, s = 4096, accuracy_thresold = None):

    criterion = torch.nn.CrossEntropyLoss()
    samples_processed = 0
    for i, (x, y) in tqdm(enumerate(trainloader)):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        output, _ = model(x.squeeze(1))
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        samples_processed += batch_size
        if samples_processed % s == 0:
            print('Iteration: %d, Loss: %f' % (i, loss.item()))
            char_acc, test_loss = evaluate(model, testloader)

            # Add accuracy check
            if accuracy_thresold is not None and char_acc >= accuracy_thresold:
                print("Accuracy reached")
                break         
            model.train()
    return samples_processed / s

        
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
            x, y = x.cuda(), y.cuda()
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
    print("Test loss mean:", total_loss / len(testloader))
    total_loss = total_loss / len(testloader)
    character_accuracy = correct_characters / total_characters
    print('Character Level Accuracy: %.2f%%' % (character_accuracy * 100))
    return character_accuracy, total_loss



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--pretrained_weights_path')
    args = parser.parse_args()
    
    dataset_name = args.dataset
    seq_length = 64
    N = 3 # Number of training runs per dataset


    train_dataset, val_dataset = get_dataset(dataset_name, train_split=0.9, seq_length=seq_length)
    vocab_size = train_dataset.vocab_size
    
    max_samples = 409600

    # Add accuracy threshold per datasets
    accuracy_thresholds = {"shakespeare": 0.6, "austen": 0.62, "dickens": 0.61}
    accuracy_thresold = accuracy_thresholds[dataset_name]


    b_128_lr = 0.001
    batch_sizes = [4, 8, 16, 32, 64, 128]

    batches_lrs = {b: b_128_lr * ((b / 128)**0.5) for b in batch_sizes } 
    
    r = np.zeros((len(batch_sizes), N))

    for idx in range(N):
        for batch_idx, batch_size in enumerate(batch_sizes):
            # Sample for replacement
            sampler = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=max_samples)
            trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=sampler)

            testloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=128, shuffle=False, drop_last = True)
            model = GPTLanguageModel(vocab_size, seq_length, n_embd=256, n_head=6, n_layer=6, dropout=0.1)

            # Load pretrained checkpoint
            if args.pretrained:
                checkpoint = torch.load(args.pretrained_weights_path, map_location = "cuda:0")
                model.load_state_dict(checkpoint["model_state_dict"])
            
            model = model.cuda()
            lr = batches_lrs[batch_size]
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
            # Evaluate the model before training
            char_acc, test_loss = evaluate(model, testloader)
            # Start training
            N_s = train(model, trainloader, testloader, optimizer, accuracy_thresold = accuracy_thresold)
            
            r[batch_idx, idx] = N_s
    r = np.mean(r, axis= 1)
    r = r / np.max(r)
    print("Batch relation vector:", r)
    
    


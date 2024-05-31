import os 
import torch
import argparse
import numpy as np
from transformer_pytorch.model import Task1transformer, Task1transformer_CP
from task1.data_preprocess import Task1_Data
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import torch.nn as nn

@torch.no_grad()
def validation(model, args):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    valid_dataset = Task1_Data(root=args.data_path, train_type="validation")
    valid_dataloader = DataLoader(valid_dataset, batch_size=1024, shuffle=False, drop_last=False)
    
    count = 0
    pbar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader))
    for step, data in pbar:
        product, customer, color, size, group, label = data
        product = product.to(device)
        customer = customer.to(device)
        color = color.to(device)
        size = size.to(device)
        group = group.to(device)
        label = label.to(device)
        
        output = model(product, customer, color, size, group, inference=True)
        count += torch.sum(output == label)
        
    validation_accuracy = count / len(valid_dataset)
    
    print("Total Number of validation set: ", len(valid_dataset))        
    print("Total correct count: ", count)
    print("Validation Accuracy: ", count / len(valid_dataset))
    
    return validation_accuracy
    
    
def train():
    parser = argparse.ArgumentParser()
    
    # Parsing argument 
    parser.add_argument("--product_vocab_size", type=int, default=58417, help="Number of products + 2")
    parser.add_argument("--customer_vocab_size", type=int, default=342041, help="Number of customers + 1")
    parser.add_argument("--color_vocab_size", type=int, default=644, help="Number of colors + 1")
    parser.add_argument("--size_vocab_size", type=int, default=31, help="Number of size + 1")
    parser.add_argument("--group_vocab_size", type=int, default=34, help="Number of groups + 1")
    
    parser.add_argument("--embedding_type", type=str, default="cc", help="Type of embedding")
    parser.add_argument("--hidden_size", type=int, default=768, help="Type of hidden embed size")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--data_path", type=str, default="./task1")

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_epoch", type=int, default=0)
    parser.add_argument("--wandb", type=int, default=0)

    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare dataset
    train_dataset = Task1_Data(root=args.data_path, train_type="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # Model preparation
    model = Task1transformer(product_vocab_size=args.product_vocab_size,
                             customer_vocab_size=args.customer_vocab_size, 
                             color_vocab_size=args.color_vocab_size,
                             size_vocab_size=args.size_vocab_size,
                             group_vocab_size=args.group_vocab_size,
                             hidden=args.hidden_size,
                             attn_heads=args.num_heads).to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    model_checkpoint = torch.load(f"./task1_checkpoint/best.pth")
    model.load_state_dict(model_checkpoint['model_state_dict'])
    # start_epoch = 0
    # if args.resume:
    #     model_checkpoint = torch.load(f"./task1_checkpoint/{args.resume_epoch}.pth")
    #     model.load_state_dict(model_checkpoint['model_state_dict'])
    #     optim.load_state_dict(model_checkpoint['optim_state_dict'])
    #     start_epoch = int(model_checkpoint['epoch'])
    
    # criterion = nn.CrossEntropyLoss()
    # best_accuracy = 0
    
    # if args.wandb:
    #     wandb.init(project='AI506_task1')

    # for epoch in range(start_epoch, args.epoch):
    #     pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    #     for step, data in pbar:
    #         model.train()
    #         product, customer, color, size, group, label = data
    #         product = product.to(device)
    #         customer = customer.to(device)
    #         color = color.to(device)
    #         size = size.to(device)
    #         group = group.to(device)
    #         label = label.to(device)
            
    #         output = model(product, customer, color, size, group)
            
    #         loss = criterion(output, label)
    #         optim.zero_grad()
    #         loss.backward()
    #         optim.step()
            
    #         if args.wandb:
    #             wandb.log({'loss': loss})
            
    #         description = f'Epoch: {epoch+1}/{args.epoch} || Step: {step+1}/{len(train_dataloader)} || Loss: {round(loss.item(), 4)}'
    #         pbar.set_description(description)
        
    validation_accuracy = validation(model, args)
    
        # if args.wandb:
        #     wandb.log({'validation_accuracy': validation_accuracy})
        
        # torch.save({
        #     'epoch': epoch,
        #     'val_accuracy': validation_accuracy,
        #     'model_state_dict': model.state_dict(),
        #     'optim_state_dict': optim.state_dict(),
        # }, f"./task1_checkpoint/{epoch}.pth")
        
        # if best_accuracy < validation_accuracy:
        #     best_accuracy = validation_accuracy
        #     torch.save({
        #         'epoch': epoch,
        #         'val_accuracy': validation_accuracy,
        #         'model_state_dict': model.state_dict(),
        #         'optim_state_dict': optim.state_dict(),
        #     }, f"./task1_checkpoint/best.pth")
            
            
if __name__ == '__main__':
    train()
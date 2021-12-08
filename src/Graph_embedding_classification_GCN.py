from pickle import load as pickle_load
from loguru import logger
from random import sample
from sklearn.metrics import f1_score,average_precision_score
from models import SimpleGCNClassifier
from pickle import dump as pkl_dump, load as pkl_load
import torch
import networkx
import random
import torch.nn as nn
import torch.nn.functional as F
import itertools
import dgl
import os
import pandas as pd


def get_data_loader(train_set,batch_size):
    trainloader = []
    for i in range(0,len(train_set),batch_size):
        end = min(i + batch_size,len(train_set))
        train_bg = dgl.batch([tup[0] for tup in train_set[i:end]])
        train_bg_labels = torch.tensor([tup[1] for tup in train_set[i:end]])
        trainloader.append((train_bg,train_bg_labels))
    return trainloader


def load_graph_and_complexes(data_folder_path):
    myGraphName = data_folder_path + "res_myGraph"
    with open(myGraphName, 'rb') as f:
        humap = pickle_load(f)
    
    train_complex_path =data_folder_path + "res_train_known_complex_nodes_list"
            
    with open(train_complex_path, 'rb') as f:
        train_complex_list = pickle_load(f)
                
    test_complex_path =  data_folder_path + "res_test_known_complex_nodes_list"
    with open(test_complex_path,'rb') as f:
        test_complex_list = pickle_load(f)            
    
    logger.info("Total number of complexes = {}",len(train_complex_list) + len(test_complex_list))
    return humap, train_complex_list, test_complex_list


def load_negatives(data_folder_path):
    with open(data_folder_path + "res_neg_train.out") as f:
        neg_train_comp_list_orig =[line.rstrip().split() for line in f.readlines()]
        
    with open(data_folder_path + "res_neg_test.out") as f:
        neg_test_comp_list_orig =[line.rstrip().split() for line in f.readlines()]
    
    logger.info("Length of negative training complexes = {}",len(neg_train_comp_list_orig))
    logger.info("Length of negative testing complexes = {}",len(neg_test_comp_list_orig))
    
    return neg_train_comp_list_orig, neg_test_comp_list_orig
    

def downsample_negatives(neg_train_comp_list_orig, neg_test_comp_list_orig, train_complex_list, test_complex_list,data_folder_path):
    try:
        with open(data_folder_path + "processed/downsampled_negative_train.pkl",'rb') as f:            
            neg_train_comp_list = pkl_dump(f)
            
        with open(data_folder_path + "processed/downsampled_negative_test.pkl",'rb') as f:
            neg_test_comp_list = pkl_dump(f)  
    except:    
        neg_train_comp_list = sample(neg_train_comp_list_orig,len(train_complex_list))
        neg_test_comp_list = sample(neg_test_comp_list_orig,len(test_complex_list))
        
        logger.info("Length of final testing complexes = {}",len(neg_train_comp_list) + len(neg_test_comp_list))
        
        with open(data_folder_path + "processed/ownsampled_negative_train.pkl",'wb') as f:            
            pkl_dump(neg_train_comp_list, f)
            
        with open(data_folder_path + "processed/downsampled_negative_test.pkl",'wb') as f:
            pkl_dump(neg_test_comp_list, f)        
    
    return neg_train_comp_list, neg_test_comp_list


def get_dgl_format_graphs(humap, train_complex_list, test_complex_list, neg_train_comp_list, neg_test_comp_list):
    logger.debug(humap.subgraph(train_complex_list[0]))
    logger.debug(dgl.from_networkx(humap.subgraph(train_complex_list[0]).to_directed(), edge_attrs=['weight']).nodes())

    train_complex_graph_list = [(dgl.from_networkx(humap.subgraph(graph).to_directed(),edge_attrs=['weight']),1) for graph in train_complex_list]
    test_complex_graph_list = [(dgl.from_networkx(humap.subgraph(graph).to_directed(),edge_attrs=['weight']),1) for graph in test_complex_list]
    
    neg_train_complex_graph_list = [(dgl.from_networkx(humap.subgraph(graph).to_directed(),edge_attrs=['weight']),0) for graph in neg_train_comp_list]
    neg_test_complex_graph_list = [(dgl.from_networkx(humap.subgraph(graph).to_directed(),edge_attrs=['weight']),0) for graph in neg_test_comp_list]
    # Identity determinable only from node names in list

    return train_complex_graph_list, test_complex_graph_list, neg_train_complex_graph_list, neg_test_complex_graph_list


def get_train_val_test_sets(train_complex_graph_list, neg_train_complex_graph_list, test_complex_graph_list, neg_test_complex_graph_list):

    train_val_set = train_complex_graph_list + neg_train_complex_graph_list
    test_set = test_complex_graph_list + neg_test_complex_graph_list
    
    
    random.shuffle(train_val_set)
    random.shuffle(test_set)
    
    
    split_pt = int(0.8*len(train_val_set))
    train_set = train_val_set[0:split_pt]
    val_set = train_val_set[split_pt:]
    
    return train_set, val_set, test_set


def model_training(model,embed, opt, embedding_size, trainloader,valloader, val_set, prev_epochs = 0,n_epochs_to_train = 10):

    for epoch in range(n_epochs_to_train):
        
        model.train()               # Set the model to train mode to allow gradients to be calculated
        
        cumulative_loss_train = 0.0 # Use this accumulate loss over an epoch and print it out at the end
                                    # to get an estimate for how well your model is performing
        
        for batched_graph, labels in trainloader:
            #features = batched_graph.ndata['attr']
            features = torch.tensor([[0 for i in range(embedding_size)] for j in range(batched_graph.number_of_nodes())])
            embedded_features = embed(features.argmax(-1))
            logits = model.forward(batched_graph, embedded_features) # Pass the features through the network to get logits
            loss = F.cross_entropy(logits, labels)  # Compute cross entropy loss. 
                                                    # HINT : Use the loss functions defined within torch.nn.functional
            
            opt.zero_grad()         # Reset gradients for the next batch, since they accumulate by default
            loss.backward()         # Backprop
            opt.step()              # Update parameters
            
            cumulative_loss_train += loss.item()
        
        with torch.no_grad():
    
            correct = 0
            model.eval()
    
            for batched_graph, labels in valloader:
                #features = batched_graph.ndata['attr'] 
                features = torch.tensor([[0 for i in range(embedding_size)] for j in range(batched_graph.number_of_nodes())])            
                embedded_features = embed(features.argmax(-1))
                logits = model.forward(batched_graph, embedded_features)    # This is the same as the training loop
    
                # Logits is a tensor comprising other tensors of 2 elements each. These 2 elements are indicative of (not
                # but not exactly equal to) the probability of the label 0 / 1 for the graph at that index.
                # The actual label for each graph will be equal to the index which has the maximum value among the 2 elements.
                # HINT : argmax
                _, predicted = torch.max(logits.data, 1)           
    
                #print(predicted)    # It is recommended to keep this to verify if your training has worked 
                                    # and the model is not simply predicting one label for every example.
    
                correct += (predicted == labels.data).sum().item()
    
            acc = 100*correct / len(val_set)
    
        #print('Validation Accuracy: %f' % (acc*100))
        # Ideally you should see the cumulative loss decrease with each epoch
        train_loss = cumulative_loss_train / len(trainloader)
        tot_epochs = prev_epochs+epoch
        logger.info("Epoch: {} Train loss: {} Validation Accuracy: {}".format(tot_epochs, train_loss ,acc))
        
    res_dict = {"Epochs": tot_epochs, "Training loss": train_loss, "Validation accuracy": acc}
    return model, res_dict, embed, opt


def model_testing(model, testloader, embed):
    # ### 7. Testing
    
    with torch.no_grad():
    
        correct = 0
        model.eval()
        
        test_predictions = []
        test_gt_lbls = []
    
        for batched_graph, labels in testloader:
            #features = batched_graph.ndata['attr'] 
            features = torch.tensor([[0 for i in range(7)] for j in range(batched_graph.number_of_nodes())])            
            embedded_features = embed(features.argmax(-1))
            logits = model.forward(batched_graph, embedded_features)    # This is the same as the training loop
    
            # Logits is a tensor comprising other tensors of 2 elements each. These 2 elements are indicative of (not
            # but not exactly equal to) the probability of the label 0 / 1 for the graph at that index.
            # The actual label for each graph will be equal to the index which has the maximum value among the 2 elements.
            # HINT : argmax
            _, predicted = torch.max(logits.data, 1)           
    
            #print(predicted)    # It is recommended to keep this to verify if your training has worked 
                                # and the model is not simply predicting one label for every example.
    
            correct += (predicted == labels.data).sum().item()
            test_predictions = test_predictions + list(predicted.data)
            test_gt_lbls = test_gt_lbls + list(labels.data)
    return correct, test_predictions, test_gt_lbls


def evaluate(test_predictions, test_gt_lbls, correct, res_dict=dict()):
    acc = 100*correct / len(test_predictions)
    f1 = f1_score(test_gt_lbls, test_predictions)
    aps = average_precision_score(test_gt_lbls, test_predictions)
    
    logger.info("Test Accuracy: {}" , acc)
    logger.info("Test F1 Score : {}", f1)
    logger.info("Test APS Score : {}", aps)
    
    res_dict["Test Accuracy"] = acc
    res_dict["Test F1 Score"] = f1
    res_dict["Test APS Score"] = aps
    
    return res_dict
    
    
def write_results(res_dict, out_dir, model, embed, opt, results_df=pd.DataFrame(),results_path='./'):
    if not os.path.exists(results_path + out_dir):
        os.mkdir(results_path + out_dir)    
        
    if len(results_df) == 0:
        results_df = pd.DataFrame(columns = res_dict.keys())
    results_df = results_df.append(pd.Series(res_dict,name = out_dir))
        
    results_df.to_csv(results_path + out_dir + '/results.csv')
    
    with open(results_path + out_dir +  '/model.pkl','wb') as f:
        pkl_dump(model,f)
        
    with open(results_path + out_dir + '/embedding.pkl','wb') as f:            
        pkl_dump(embed, f)
        
    with open(results_path + out_dir + '/optimizer.pkl','wb') as f:
        pkl_dump(opt, f)        
    

def main():
    results_path = '../results/'
    out_dir = 'GCN_downsampled_negatives'
    data_folder_path = '../data/'
    downsample_negatives_flag = 1
    
    # out_dir = 'GCN'
    # downsample_negatives_flag = 0
    n_epochs_to_train = 10
    
        # Check for GPU
    if torch.cuda.is_available():
        logger.info('Default GPU Device : {}'.format(torch.cuda.get_device_name(0)))
        device = torch.device('cuda')
    else:
        logger.info('No GPU available')
        device = torch.device('cpu')
        
    logger.info(device)

    humap, train_complex_list, test_complex_list = load_graph_and_complexes(data_folder_path)
    
    neg_train_comp_list, neg_test_comp_list = load_negatives(data_folder_path)
    
    if downsample_negatives_flag:
        neg_train_comp_list, neg_test_comp_list = downsample_negatives(neg_train_comp_list, neg_test_comp_list, train_complex_list, test_complex_list,data_folder_path)
    
    train_complex_graph_list, test_complex_graph_list, neg_train_complex_graph_list, neg_test_complex_graph_list = get_dgl_format_graphs(humap, train_complex_list, test_complex_list, neg_train_comp_list, neg_test_comp_list)
    train_set, val_set, test_set = get_train_val_test_sets(train_complex_graph_list, neg_train_complex_graph_list, test_complex_graph_list, neg_test_complex_graph_list)
    
    
    # Get batched data sets
    batch_size = 32
    
    trainloader = get_data_loader(train_set,batch_size)
    valloader = get_data_loader(val_set,batch_size)
    testloader = get_data_loader(test_set,batch_size)
    
    # ### 5. Training
    embedding_size = 7 # Embedding of size 7 chosen randomly
    try:
        with open(results_path + out_dir + '/model.pkl','rb') as f:
            model = pkl_load(f)
        with open(results_path + out_dir + '/embedding.pkl','rb') as f:            
            embed = pkl_load(f)
        with open(results_path + out_dir + '/optimizer.pkl','rb') as f:
            opt = pkl_load(f)
            
        results_df = pd.read_csv(results_path + out_dir + '/results.csv', index_col = 0)
        prev_epochs = results_df['Epochs'][-1]
        
    except: # train new model
        model = SimpleGCNClassifier(5, 25, 2) 
        embed = nn.Embedding(embedding_size, 5)    
        opt = torch.optim.Adam(itertools.chain(model.parameters(), embed.parameters()))
        results_df = pd.DataFrame()
        prev_epochs = 0
    
    model, res_dict, embed, opt = model_training(model,embed, opt, embedding_size, trainloader,valloader, val_set, prev_epochs,n_epochs_to_train)
    correct, test_predictions, test_gt_lbls = model_testing(model, testloader, embed)
    res_dict = evaluate(test_predictions, test_gt_lbls, correct, res_dict)
    
    write_results(res_dict, out_dir, model, embed, opt, results_df,results_path)
    
    
if __name__ == '__main__':
    main()

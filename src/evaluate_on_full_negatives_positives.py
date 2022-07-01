# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 19:12:36 2021

@author: Meghana
"""

from Graph_embedding_classification_GCN import *

data_folder_path = '../data/'
#results_path = '../results/'
results_path = '../results/20k_epochs/'


humap, train_complex_list, test_complex_list = load_graph_and_complexes(data_folder_path)

neg_train_comp_list, neg_test_comp_list = load_negatives(data_folder_path)



embedding_size = 7 # 7 chosen randomly, also the size of features
out_layer_size = 2 # Equal to the number of classes

embedding_options = [0,1]
batch_size = 32

results_df = pd.DataFrame()
res_dict = dict()

for embedding_layer_flag in embedding_options:
    if embedding_layer_flag:
        input_layer_size = 5
    else:
        input_layer_size = embedding_size
        
    models = [OneConvTwoClassiLayerGCN(input_layer_size, 25, out_layer_size), 
              TwoConvThreeClassiLayerGCN(input_layer_size, 25, 27, out_layer_size),
              TwoConvThreeClassiLayerGCN(input_layer_size, 25, 30, out_layer_size),
              TwoConvFourClassiLayerGCN(input_layer_size, 25, 12, out_layer_size),
              OneConvSAGE(input_layer_size, out_layer_size),
              TwoConvSAGE(input_layer_size, 25, out_layer_size)
              ]        
    for model in models:
        out_dir = 'emb' +str(embedding_layer_flag) + str(model)[:12]
        try:
            model, embed, opt, results_old_df = load_model(results_path,out_dir,embedding_layer_flag)
            train_complex_graph_list, test_complex_graph_list, neg_train_complex_graph_list, neg_test_complex_graph_list = get_dgl_format_graphs(humap, train_complex_list, test_complex_list, neg_train_comp_list, neg_test_comp_list)
            train_set, val_set, test_set = get_train_val_test_sets(train_complex_graph_list, neg_train_complex_graph_list, test_complex_graph_list, neg_test_complex_graph_list)
            
            
            # Get batched data sets
            
            trainloader = get_data_loader(train_set,batch_size)
            valloader = get_data_loader(val_set,batch_size)
            testloader = get_data_loader(test_set,batch_size)
           
            correct, test_predictions, test_gt_lbls = model_testing(model, testloader, embed)
            res_dict = evaluate(test_predictions, test_gt_lbls, correct, res_dict)

            if len(results_df) == 0:
                results_df = pd.DataFrame(columns = res_dict.keys())    
            results_df = results_df.append(pd.Series(res_dict,name = out_dir))

        except Exception as e:
            logger.error(str(e))
            
results_df.to_csv(results_path + '/results_full_negatives_test.csv')


    
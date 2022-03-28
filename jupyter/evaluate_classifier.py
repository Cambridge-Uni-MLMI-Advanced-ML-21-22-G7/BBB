import torch
import numpy as np

def evaluate_classifier(net, X_val):
    correct = 0
    total = 0

    # Logging
    dict = {}
    dict['labels'] = []
    dict['probs'] = {}
    dict['preds'] = {}
    dict['preds']['incorrect'] = []
    dict['preds']['correct'] = []
    dict['preds']['all'] = []

    dict['probs']['all'] = []
    dict['probs']['max'] = []

    dict['probs']['incorrect'] = {}
    dict['probs']['correct'] = {}
    dict['probs']['incorrect']['all'] = []
    dict['probs']['correct']['all'] = []
    dict['probs']['incorrect']['max'] = []
    dict['probs']['correct']['max'] = []

    # Eval & save results appropriately
    with torch.no_grad():
        for inputs, labels in X_val:
            # inputs = inputs.to(DEVICE)
            # labels = labels.to(DEVICE)
            
            preds, probs = net.predict(inputs)

            total += net.batch_size
            correct += (labels == preds).sum()

            # create two arrays;
            preds = preds.cpu().detach().numpy()
            probs = probs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            correct_filter = np.where(labels == preds)
            incorrect_filter = np.where(labels != preds)

            correct_preds = preds[correct_filter]
            correct_probs = probs[correct_filter]
            
            incorrect_preds = preds[incorrect_filter]
            incorrect_probs = probs[incorrect_filter]


            dict['labels'].extend(labels)
            dict['probs']['all'].extend(probs) 
            dict['preds']['all'].extend(preds)
            
            dict['probs']['max'].append(np.max(probs, axis=1))
            
            dict['preds']['incorrect'].extend(incorrect_preds)
            dict['preds']['correct'].extend(correct_preds)
            
            dict['probs']['incorrect']['all'].extend(incorrect_probs)
            dict['probs']['correct']['all'].extend(correct_probs)

    eval_score = correct / total

    dict['eval_score'] = eval_score
    return dict
import os
import pdb
from tqdm import tqdm
import json
import numpy as np
from operator import getitem
from neat_src1 import ann
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap, value_and_grad
from neat_src1.utils import argsort

def train(neat, samples, targets, lr=0.1, batch_size=5, backprop=False):
    """Runs each neat ind on the dataset and sets fitness to the score

    Args: 
        -neat       (neat obj, see neat.py) - A population of individuals
        -samples    (list-like) - shape [Ndatapoints x 2] giving the [x.y] of each point
        -targets    (list-like) - shape [Ndatapoints x 1] giving the class label (0 or 1)
        -lr         (float) - learning rate, specified in config.json
        -batch_size (int)   - batch size, set in config.json
        -backprop   (bool)  - whether or not to perform backprop. In rapid evolution 
                                .. phases this is set to False
    """

    targets = np.expand_dims(np.array(targets),1)
    for ind in neat.pop:
        total_loss = 0
        dw = 0 
        broken = False
        orig_wMat = np.copy(ind.wMat)
        for i, (x,t) in enumerate(zip(samples, targets)):
            loss_val, W_grad = get_gradient_value(jnp.array(ind.wMat), jnp.array(ind.aVec),
                                                  inPattern=jnp.array(x), label=jnp.array(t), 
                                                  nInput=2, nOutput=1)
            
            if backprop:
                # Pseudo-stochastic gradient descent update
                dw += W_grad
                if (i % batch_size) == 0:        
                    ind.wMat -= np.array(lr * dw) 
                    dw = 0
            total_loss += loss_val

        if backprop:
            ind.conn, broken = updateconn(orig_wMat, ind.wMat, ind.conn)

        ind.fitness = -float(total_loss / len(samples))
        if broken: 
            neat.pop.remove(ind)
            neat.p['popSize'] -= 1
            print(f"Pop size is now {len(neat.pop)}")
    
    aveloss = sum([-ind.fitness for ind in neat.pop]) / len(neat.pop)
    best_ind = neat.pop[argsort([ind.fitness for ind in neat.pop])[-1]]
    print(f"Loss is {aveloss}")
    print(f"Best loss is {-best_ind.fitness}")
    
    return aveloss
    

def eval(ind, samples, targets): 
    """Given an ind and the dataset, evaluate the individual
    Args:
        -samples    (list-like) - shape [Ndatapoints x 2] giving the [x.y] of each point
        -targets    (list-like) - shape [Ndatapoints x 1] giving the class label (0 or 1)
    """
    preds, logits = [], []
    score = 0

    for x, t in zip(np.stack(samples),targets): 
        out = predict(ind.wMat, ind.aVec, nInput=2, nOutput=1, inPattern=x)
        logit = sigmoid(out)
        pred = int(round(logit)[0])
        preds.append(pred)
        logits.append(float(logit[-1]))

        if pred == t:
            score += 1 
    print(f"Accuracy is {score / len(samples)}")
    return preds, score, logits

def updateconn(orig_wMat, newMat, conn):
    """We take the updated wMat and update the conn[3,:]
    We do this by finding the indices in wMat that match the conn weights
    and replace conn weights with gradient updated weight

    Args: 
        -orig_wMat (list-like) - weight matrix, one row and column for each node
                    [N X N]    / rows: connection from; cols: connection to;
        -newMat    (list-like) - weight matrix after gradient update, one row and column 
                                 ..for each node
                    [N X N]    / rows: connection from; cols: connection to;
        -conn  -  see Ind.py
    """
    orig_wMat = np.array(orig_wMat)
    newMat = np.array(newMat)
    conn = np.array(conn)
    nonzeros = np.nonzero(orig_wMat)
    new_weights = newMat[nonzeros]
    broken = False

    for col in conn.T:
        # If inactive, skip
        if col[4] == 0:
            continue
    
        # We round them to 3 decimal places to compare (originally different datatypes)
        idx = np.where(round(col[3],3) == np.round(orig_wMat[nonzeros],3))[0]
        try:
            col[3] = new_weights[idx][-1]
        except:
            # Sometimes there are rounding errors in array conversion.  We now round to 
            # .. two decimals.  If this doesnt work, we eliminate the individual
            idx = np.where(round(col[3],2) == np.round(orig_wMat[nonzeros],2))[0]
            try:
                col[3] = new_weights[idx][-1]
            except: 
                print("Error finding conn corresponding to wMat.  Removing indivdual")
                broken = True
    return conn, broken 
    
def plot_results(inputs, targets, preds, losses, xor:bool=False):
    """Plots a scatter plot of predictions vs. ground truths 
    Plots a Loss vs. Epoch curve over training history 

    Args:
        -inputs  (list-like) - shape [Ndatapoints x 2] giving the [x.y] of each point
        -targets (list-like) - shape [Ndatapoints x 1] giving the class label (0 or 1)
        -preds   (list-like) - shape [Ndatapoints x 1] giving the predicted label (after rounding)
        -losses  (list-like) - shape [Nepochs, ] giving the loss for each epoch / evolution 
        -circle  (bool)      - boolean indicating whether it is the xor dataset or not
    """
    import matplotlib.pyplot as plt 
    
    targets = np.array(targets)
    preds = np.array(preds)
    inputs = np.array(inputs)

    wrongs = np.where(targets != preds)
    onesidxs = (targets == preds) * (targets == 0)
    zidxs    = (targets == preds) * (targets == 1)

    plt.scatter(inputs[:,0][wrongs], inputs[:,1][wrongs], color='r')
    plt.scatter(inputs[:,0][onesidxs], inputs[:,1][onesidxs], color='b')
    plt.scatter(inputs[:,0][zidxs], inputs[:,1][zidxs], color='g')
    plt.ylim([-1,1])
    plt.xlim([-1,1])
    if xor: 
        plt.axhline(y=0.0, color='k', linestyle='-')
        plt.axvline(x=0.0, color='k', linestyle='-')
    else:
        plt.grid()

    plt.legend(['Incorrect'])
    plt.title(f"Accuracy = {(preds==targets).sum()/len(targets)}")
    plt.show()

    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.title("Loss vs. Epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.grid()
    plt.show()

def BCEloss(wMat, aVec, nInput, nOutput, inPattern, label):
    """Calculate Binary Cross Entropy loss for each sample. 
    """
    h = predict(wMat, aVec, nInput, nOutput, inPattern)
    sigout = sigmoid(h)
    loss = label * jnp.log(sigout + 1e-7) + (1-label) * jnp.log(1-sigout + 1e-7)
    # loss = 10 * (label - sigout)**2
    return -3*jnp.sum(loss)

def get_gradient_value(wMat, aVec, inPattern, label, nInput, nOutput): 
    loss_val, W_grad = value_and_grad(BCEloss)(wMat, aVec, 
                                               jnp.array(nInput), jnp.array(nOutput), 
                                               inPattern, label)
    return loss_val, W_grad

def predict(wMat, aVec, nInput, nOutput, inPattern):
    h = ann.act(jnp.array(wMat), jnp.array(aVec), nInput=nInput, nOutput=nOutput, 
                 inPattern=jnp.array(inPattern))
    return jnp.array(h[0])

@jax.jit
def cross_entropy_loss(logits, labels):
    sigout = sigmoid(logits)
    loss = labels * jnp.log(sigout + 1e-7) + (1-labels) * jnp.log(1-sigout + 1e-7)
    return 10*loss.sum() / len(labels)

def sigmoid(z): 
    return 0.5 * (jnp.tanh(z / 2) + 1)

def dsig_dx(x): 
    return sigmoid(x) * (1 - sigmoid(x))

def dloss_dsig(x, y): 
    return (sigmoid(x) - y) / (sigmoid(x) * (1 - sigmoid(x))) 

def dloss_dw(logits, labels, inputs):
    dLdw = (sigmoid(logits) - labels).sum()/len(labels) * inputs #(1,1) * (100,2)
    return dLdw

class AdamOptim():
    def __init__(self, eta=0.5, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.t = 1
    def update(self, w, dw):
        ## dw, db are from current minibatch
        ## momentum beta 1
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
    
        ## rms beta 2
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
    
        ## bias correction
        m_dw_corr = self.m_dw/(1-self.beta1**self.t)
        v_dw_corr = self.v_dw/(1-self.beta2**self.t)
        
        w = w - self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        return w

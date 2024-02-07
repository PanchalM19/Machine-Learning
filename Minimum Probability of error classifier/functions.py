import numpy as np
from sys import float_info
from scipy.stats import multivariate_normal, norm

def generate_gaussian_pdf(samples, dimensions, pdf):
    X = np.zeros((samples, dimensions))
    labels = np.zeros(samples)
    #calculate thresholds
    thresholds = np.cumsum(pdf['priors'])
    thresholds = np.insert(thresholds, 0, 0)
    #randomly make samples based on priors
    for l in range(len(pdf['priors'])):
        indices = np.where((thresholds[l] <= np.random.rand(samples)) & (np.random.rand(samples) <= thresholds[l+1]))[0]
        Nl = len(indices)
        labels[indices] = l
        if dimensions == 1:
            X[indices, 0] = norm.rvs(pdf['mu'][l], pdf['sigma'][l], Nl)
        else:
            X[indices, :] = multivariate_normal.rvs(pdf['mu'][l], pdf['sigma'][l], size=Nl)
    return X, labels


# Generate ROC curve samples

def estimate_roc(discriminant, labels):
    N_labels = np.array([np.sum(labels == 0), np.sum(labels == 1)])
    score = np.sort(discriminant)
    gammas = np.concatenate(([score[0] - np.finfo(float).eps], score, [score[-1]+ np.finfo(float).eps]))
    #decision label
    decisions = [discriminant >= g for g in gammas]
                
    #FPR & TPR
    ind10 = [np.argwhere((d == 1) & (labels == 0)) for d in decisions]
    p10 = [len(inds) / N_labels[0] for inds in ind10]
    ind11 = [np.argwhere((d == 1) & (labels == 1)) for d in decisions]
    p11 = [len(inds) / N_labels[1] for inds in ind11]
    #output
    roc = {'p10': np.array(p10), 'p11': np.array(p11)}
    return roc, gammas

def get_metrics(predictions, labels):
    metrics = {}

    #true negative probability rate
    ind_00 = np.argwhere((predictions == 0) & (labels == 0))
    metrics['TNR'] = len(ind_00) /(sum(labels == 0))
    #false positive probability rate
    ind_10 = np.argwhere((predictions == 1) & (labels == 0))
    metrics['FPR'] = len(ind_10) /(sum(labels == 0))
    #false negative probability rate
    ind_01 = np.argwhere((predictions == 0) & (labels == 1))
    metrics['FNR'] = len(ind_01) / sum(labels == 1)
    #true positive probability rate
    ind_11 = np.argwhere((predictions == 1) & (labels == 1))
    metrics['TPR'] = len(ind_11) / sum(labels == 1)
    return metrics

def erm_classification(X, Lambda, params, C):
    likelihoods = np.array([multivariate_normal.pdf(X, params['mu'][i], params['sigma'][i]) for i in range(C)])
    priors = np.diag(params['priors'])
    posteriors = priors.dot(likelihoods)
    risk = Lambda.dot(posteriors)
    return np.argmin(risk, axis=0)




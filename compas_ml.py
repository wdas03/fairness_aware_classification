from scipy.spatial.distance import cdist
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.metrics import confusion_matrix, recall_score

# knn similarity scores
def normed_distance(x1, x2, p=2.): 
    if p == 'inf':
        return cdist(x1, x2, metric='chebyshev')
    
    return cdist(x1, x2, metric='minkowski', p=p)

# pdf of multivariate guassian distribution
def multivariate_gaussian_pdf(x, mean, covariance_matrix):
    return (1 / (np.power(np.pi, len(x) / 2) * np.sqrt(np.linalg.det(covariance_matrix)))) * (np.exp(-(0.5) * np.dot(np.dot((x - mean).T, np.linalg.inv(covariance_matrix)), (x - mean))))

# Socially/fairness aware binary classifier
class BinaryClassifier():
    def __init__(self):
        self.train_data = pd.read_csv('compas_dataset/propublicaTrain.csv')
        self.test_data = pd.read_csv('compas_dataset/propublicaTest.csv')

        self.X_train = self.train_data.drop('two_year_recid', axis=1).to_numpy()
        self.X_test = self.test_data.drop('two_year_recid', axis=1).to_numpy()

        self.y_train = self.train_data['two_year_recid'].to_numpy()
        self.y_test = self.test_data['two_year_recid'].to_numpy()

        self.X_train_race = self.train_data['race'].to_numpy()
        self.X_test_race = self.test_data['race'].to_numpy()

        self.num_classes = len(set(self.y_train))

    def split_data(self):
        self.X_train = self.train_data.drop('two_year_recid', axis=1).to_numpy()
        self.X_test = self.test_data.drop('two_year_recid', axis=1).to_numpy()

        self.y_train = self.train_data['two_year_recid'].to_numpy()
        self.y_test = self.test_data['two_year_recid'].to_numpy()

        self.num_classes = len(set(self.y_train))

    def get_predictions(self):
        raise NotImplementedError("Must override method.")
    
    def get_accuracy(self):
        return np.mean(self.y_pred == self.y_test)

    def get_sensitivity(self):
        return recall_score(self.y_test, self.y_pred)
    
    def training_size_performance(self):
        accuracies, sensitivities = [], []
        for p in np.arange(.1, 1.1, .1):
            self.train_data = pd.read_csv('compas_dataset/propublicaTrain.csv')
            self.train_data = self.train_data.sample(frac = p, random_state=0)

            #print(p)
            self.split_data()
            self.get_predictions()

            accuracies.append((int(len(self.train_data) * p), self.get_accuracy()))
            sensitivities.append((int(len(self.train_data) * p), self.get_sensitivity()))
            
        return accuracies, sensitivities
    
    def demographic_parity(self):
        #print(self.y_pred)
        #print(self.X_test_race)
        race_0_pred = self.y_pred[self.X_test_race == 0]
        race_1_pred = self.y_pred[self.X_test_race == 1]

        #race_0_true = self.y_test[self.y_test[:, 2] == 0]
        #race_1_true = self.y_test[self.y_test[:, 2] == 1]

        race_0_dp = np.count_nonzero(race_0_pred == 1) / len(race_0_pred)
        race_1_dp = np.count_nonzero(race_1_pred == 1) / len(race_1_pred)

        return (race_0_dp, race_1_dp)
    
    def equalized_odds(self):
        race_0_pred = self.y_pred[self.X_test_race == 0]
        race_1_pred = self.y_pred[self.X_test_race == 1]

        race_0_true = self.y_test[self.X_test_race == 0]
        race_1_true = self.y_test[self.X_test_race == 1]

        race_0_cm = confusion_matrix(race_0_true, race_0_pred)
        race_1_cm = confusion_matrix(race_1_true, race_1_pred)

        race_0_tn, race_0_fp, race_0_fn, race_0_tp = race_0_cm.ravel()
        race_1_tn, race_1_fp, race_1_fn, race_1_tp = race_1_cm.ravel()

        race_0_tpr, race_0_tnr = (race_0_tp / (race_0_tp + race_0_fn)), (race_0_tn / (race_0_tn + race_0_fp))
        race_1_tpr, race_1_tnr = (race_1_tp / (race_1_tp + race_1_fn)), (race_1_tn / (race_1_tn + race_1_fp))

        return (race_0_tpr, race_0_tnr, race_1_tpr, race_1_tnr)

    def predictive_parity(self):
        race_0_pred = self.y_pred[self.X_test_race == 0]
        race_1_pred = self.y_pred[self.X_test_race == 1]

        race_0_true = self.y_test[self.X_test_race == 0]
        race_1_true = self.y_test[self.X_test_race == 1]

        race_0_cm = confusion_matrix(race_0_true, race_0_pred)
        race_1_cm = confusion_matrix(race_1_true, race_1_pred)

        race_0_tn, race_0_fp, race_0_fn, race_0_tp = race_0_cm.ravel()
        race_1_tn, race_1_fp, race_1_fn, race_1_tp = race_1_cm.ravel()

        race_0_ppv, race_0_npv = (race_0_tp / (race_0_tp + race_0_fp)), (race_0_tn / (race_0_tn + race_0_fn))
        race_1_ppv, race_1_npv = (race_1_tp / (race_1_tp + race_1_fp)), (race_1_tn / (race_1_tn + race_1_fn))

        return (race_0_ppv, race_0_npv, race_1_ppv, race_1_npv)


class MLE_classifier(BinaryClassifier):
    def __init__(self):
        super().__init__()
    
    def get_predictions(self):
        # Assign class priors (for 0 and 1)
        class_priors = {}
        for i in range(self.num_classes):
            class_priors[i] = np.count_nonzero(self.y_train == i) / len(self.y_train)
        
        # Calculate MLE estimates for mean and covariance of each class
        class_means = {}
        class_covariances = {}
        for i in range(self.num_classes):
            class_means[i] = np.mean(self.X_train[self.y_train == i], axis=0)
            class_covariances[i] = np.cov(self.X_train[self.y_train == i], rowvar=False)

        # Evaluate model
        class_probs = []
        for x_test in self.X_test:
            probs = []
            for i in range(self.num_classes):
                class_conditional = multivariate_gaussian_pdf(x_test, class_means[i], class_covariances[i] + (1e-4 * np.identity(len(x_test))))
                probs.append(class_priors[i] * class_conditional)

            class_probs.append(probs)
        
        self.y_pred = np.argmax(class_probs, axis=1)
        return self.y_pred
    
class KNN_classifier(BinaryClassifier):
    def __init__(self, k=85, metric='L2'):
        super().__init__()
        self.k = k
        self.metric = metric

    def get_predictions(self):
        # Normalize data (min-max)
        min_vals = np.min(self.X_train, axis=0)
        max_vals = np.max(self.X_train, axis=0)
        
        # Apply min-max scaling to train and test data
        X_train_normalized = (self.X_train - min_vals) / (max_vals - min_vals)
        X_test_normalized = (self.X_test - min_vals) / (max_vals - min_vals)

        # Distance metric
        p = 2
        if self.metric == 'L1':
            p = 1
        elif self.metric == 'L2':
            p = 2
        elif self.metric == 'L-inf':
            p = 'inf'

        self.y_pred = []
        for x in X_test_normalized:
            k_neighbors = None

            dists = normed_distance(x.reshape(1, -1), X_train_normalized, p)[0]
            k_neighbor_indices = np.argsort(dists)[:self.k]
            k_neighbor_labels = self.y_train[k_neighbor_indices]

            self.y_pred.append(np.bincount(k_neighbor_labels).argmax())
        
        self.y_pred = np.array(self.y_pred)
        
        return np.array(self.y_pred)

class NB_classifier(BinaryClassifier):
    def __init__(self):
        super().__init__()

        self.train_data['age'] = pd.qcut(self.train_data['age'], 10)
        
        self.X_train = self.train_data.drop('two_year_recid', axis=1).to_numpy()
        self.X_test = self.test_data.drop('two_year_recid', axis=1).to_numpy()

        self.y_train = self.train_data['two_year_recid'].to_numpy()
        self.y_test = self.test_data['two_year_recid'].to_numpy()

        self.num_classes = len(set(self.y_train))
    
    def get_predictions(self):
        class_priors = {}
        class_counts = {}
        for i in range(self.num_classes):
            class_priors[i] = np.count_nonzero(self.y_train == i) / len(self.y_train)
            class_counts[i] = np.count_nonzero(self.y_train == i)

        # Use Laplace MAP estimate for class conditional densities
        y_pred = []

        class_probs = []
        for x_test in self.X_test:
            probs = []
            for i in range(self.num_classes):
                class_conditional = 0
                for idx, x_feature in enumerate(x_test):
                    X_train_features = self.X_train[:, idx]

                    class_conditional += math.log((np.count_nonzero(X_train_features[self.y_train == i] == x_feature) + 1) / (class_counts[i] + 2))
                
                probs.append(math.log(class_priors[i]) + class_conditional)

            class_probs.append(probs)
        
        self.y_pred = np.argmax(class_probs, axis=1)
        
        return self.y_pred

def test_k(value_range=range(5, 500, 20), metric='L1'):
    accuracies = []
    for i in value_range:
        knn_clf = KNN_classifier(k=i, metric=metric)
        knn_clf.get_predictions()
        print(metric, i, knn_clf.get_accuracy())

        accuracies.append((i, knn_clf.get_accuracy()))
    
    return np.array(accuracies)

def plot_knn_accuracies(value_range=range(5, 4000, 50)):
    knn_l1 = test_k(value_range, metric='L1')
    knn_l2 = test_k(value_range, metric='L2')
    knn_l_inf = test_k(value_range, metric='L-inf')

    plt.plot(knn_l1[:, 0], knn_l1[:, 1], label='L1')
    plt.plot(knn_l2[:, 0], knn_l2[:, 1], label='L2')
    plt.plot(knn_l_inf[:, 0], knn_l_inf[:, 1], label='L-inf')

    plt.legend()
    plt.title("kNN Test Accuracy")
    plt.xlabel("k")
    plt.ylabel("Test Set Accuracy")

def plot_demographic_parity():
    mle_clf = MLE_classifier()
    mle_clf.get_predictions()
    mle_dp = mle_clf.demographic_parity()

    knn_clf = KNN_classifier()
    knn_clf.get_predictions()
    knn_dp = knn_clf.demographic_parity()

    nb_clf = NB_classifier()
    nb_clf.get_predictions()
    nb_dp = nb_clf.demographic_parity()

    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))

    # set height of bar
    race_0_dp = [d[0] for d in [mle_dp, knn_dp, nb_dp]]
    race_1_dp = [d[1] for d in [mle_dp, knn_dp, nb_dp]]

    # Set position of bar on X axis
    br1 = np.arange(len(race_0_dp))
    br2 = [x + barWidth for x in br1]

    # Make the plot
    plt.bar(br1, race_0_dp, color ='r', width = barWidth,
            edgecolor ='grey', label ='Race 0')
    plt.bar(br2, race_1_dp, color ='g', width = barWidth,
            edgecolor ='grey', label ='Race 1')

    # Adding Xticks
    plt.title("Demographic Parity Across Race Attribute")
    plt.xlabel('Classifier', fontweight ='bold', fontsize = 15)
    plt.ylabel('Positive Rate Metric', fontweight ='bold', fontsize = 15)
    plt.xticks([r + .12 for r in range(len(race_0_dp))],
            ['MLE', 'kNN', 'Naive Bayes'])

    plt.legend()
    #plt.ylim(0, 1)

    plt.show()

def plot_eo():
        mle_clf = MLE_classifier()
        mle_clf.get_predictions()
        mle_eo = mle_clf.equalized_odds()

        knn_clf = KNN_classifier()
        knn_clf.get_predictions()
        knn_eo = knn_clf.equalized_odds()

        nb_clf = NB_classifier()
        nb_clf.get_predictions()
        nb_eo = nb_clf.equalized_odds()

        # set width of bar
        barWidth = 0.1
        fig = plt.subplots(figsize =(15, 10))
        
        # set height of bar
        race_0_tpr = [d[0] for d in [mle_eo, knn_eo, nb_eo]]
        race_0_tnr = [d[1] for d in [mle_eo, knn_eo, nb_eo]]
        race_1_tpr = [d[2] for d in [mle_eo, knn_eo, nb_eo]]
        race_1_tnr = [d[3] for d in [mle_eo, knn_eo, nb_eo]]
        
        # Set position of bar on X axis
        br1 = np.arange(len(race_0_tpr))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]
        
        # Make the plot
        plt.bar(br1, race_0_tpr, color ='r', width = barWidth,
                edgecolor ='grey', label ='Race 0 TPR')
        plt.bar(br2, race_1_tpr, color ='g', width = barWidth,
                edgecolor ='grey', label ='Race 1 TPR')
        plt.bar(br3, race_0_tnr, color ='b', width = barWidth,
                edgecolor ='grey', label ='Race 0 TNR')
        plt.bar(br4, race_1_tnr, color ='y', width = barWidth,
                edgecolor ='grey', label ='Race 1 TNR')
        
        # Adding Xticks
        plt.title("Equalized Odds (TPR/TPN) Across Race Attribute")
        plt.xlabel('Classifier', fontweight ='bold', fontsize = 15)
        plt.ylabel('TPR/TNR Metrics', fontweight ='bold', fontsize = 15)
        plt.xticks([r + .15 for r in range(len(race_0_tpr))],
                ['MLE', 'kNN', 'Naive Bayes'])
        
        plt.legend()
        #plt.ylim(0, 1)

        plt.show()

def plot_pp():
        mle_clf = MLE_classifier()
        mle_clf.get_predictions()
        mle_pp = mle_clf.predictive_parity()

        knn_clf = KNN_classifier()
        knn_clf.get_predictions()
        knn_pp = knn_clf.predictive_parity()

        nb_clf = NB_classifier()
        nb_clf.get_predictions()
        nb_pp = nb_clf.predictive_parity()

        # set width of bar
        barWidth = 0.1
        fig = plt.subplots(figsize =(15, 10))
        
        # set height of bar
        race_0_ppv = [d[0] for d in [mle_pp, knn_pp, nb_pp]]
        race_0_npv = [d[1] for d in [mle_pp, knn_pp, nb_pp]]
        race_1_ppv = [d[2] for d in [mle_pp, knn_pp, nb_pp]]
        race_1_npv = [d[3] for d in [mle_pp, knn_pp, nb_pp]]
        
        # Set position of bar on X axis
        br1 = np.arange(len(race_0_ppv))
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]
        
        # Make the plot
        plt.bar(br1, race_0_ppv, color ='r', width = barWidth,
                edgecolor ='grey', label ='Race 0 PPV')
        plt.bar(br2, race_1_ppv, color ='g', width = barWidth,
                edgecolor ='grey', label ='Race 1 PPV')
        plt.bar(br3, race_0_npv, color ='b', width = barWidth,
                edgecolor ='grey', label ='Race 0 NPV')
        plt.bar(br4, race_1_npv, color ='y', width = barWidth,
                edgecolor ='grey', label ='Race 1 NPV')
        
        # Adding Xticks
        plt.title("Predictive Parity (PPV/NPV) Across Race Attribute")
        plt.xlabel('Classifier', fontweight ='bold', fontsize = 15)
        plt.ylabel('PPV/NPV Metrics', fontweight ='bold', fontsize = 15)
        plt.xticks([r + .15 for r in range(len(race_0_ppv))],
                ['MLE', 'kNN', 'Naive Bayes'])
        
        plt.legend()
        #plt.ylim(0, 1)

        plt.show()

def plot_acc_recall():
    mle_clf = MLE_classifier()
    (mle_metrics, mle_recall) = np.array(mle_clf.training_size_performance())

    knn_clf = KNN_classifier()
    (knn_metrics, knn_recall) = np.array(knn_clf.training_size_performance())

    nb_clf = NB_classifier()
    (nb_metrics, nb_recall) = np.array(nb_clf.training_size_performance())

    plt.plot(mle_metrics[:, 0], mle_metrics[:, 1], label='MLE-based Model')
    plt.plot(knn_metrics[:, 0], knn_metrics[:, 1], label='kNN')
    plt.plot(nb_metrics[:, 0], nb_metrics[:, 1], label='Naive Bayes')

    plt.legend()

    plt.title("Training Set Size vs Test Set Accuracy")
    plt.xlabel("Training Set Size")
    plt.ylabel("Test Set Accuracy")

    plt.figure()

    plt.plot(mle_recall[:, 0], mle_recall[:, 1], label='MLE-based Model')
    plt.plot(knn_recall[:, 0], knn_recall[:, 1], label='kNN')
    plt.plot(nb_recall[:, 0], nb_recall[:, 1], label='Naive Bayes')

    plt.legend()

    plt.title("Training Set Size vs Test Set Sensitivity")
    plt.xlabel("Training Set Size")
    plt.ylabel("Test Set Sensitivity")

    plt.figure()

if __name__ == "__main__":
    knn = KNN_classifier()
    knn.get_predictions()
    print("kNN -> accuracy: {:0.3f}, sensitivity: {:0.3f}".format(knn.get_accuracy(), knn.get_sensitivity()))

    mle = MLE_classifier()
    mle.get_predictions()
    print("mle -> accuracy: {:0.3f}, sensitivity: {:0.3f}".format(mle.get_accuracy(), mle.get_sensitivity()))

    nb = NB_classifier()
    nb.get_predictions()
    print("NB -> accuracy: {:0.3f}, sensitivity: {:0.3f}".format(nb.get_accuracy(), nb.get_sensitivity()))

    plot_knn_accuracies()
    plot_demographic_parity()
    plot_eo()
    plot_pp()
    plot_acc_recall()
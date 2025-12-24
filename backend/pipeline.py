import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def upload_dataset(file_path):
    """
    load the dataset from csv file
    """
    if file_path.endswith('.xlsx'):
        df= pd.read_excel(file_path)

    elif file_path.endswith('.csv'):
        df= pd.read_csv(file_path)

    elif file_path.endswith('xls'):
        df= pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
        
    return df

# i gotta allow standardization [standardscaler]
#  and normalization [minmaxscaler]

def infer_dataset(df):
    """display basic dataset info"""
    print("Dataset Shape:", df.shape)
    print("Dataset Columns:", df.columns.tolist())
    print("Your dataset looks like this:\n", df.sample(7))
    print("Wanna See the number of classes in your y?\n" )
    p= df[df.columns[-1]]
    for cls in np.unique():
        count= np.sum(p==cls)
        percentage= count/len(p)*100
        print(f"number of samples of {cls} and percentage is : {count}, {percentage} ")
    
    print("lets figure out if the dataset is balanced/imbalanced !\n")
    counts= [np.sum(p==cls) for cls in np.unique(p)]
    if (max(counts)/min(counts))<1.5:
        bal=0
        print("The dataset is fairly balanced! yip!")
    else:
        bal=1
        print("its a little imbalanced, but we will manage.")
    
    return bal


def dataset_divide(df, method):
    X= df.drop(df.columns[-1], axis=1)
    y= df[df.columns[-1]]
    categorical_cols= X.select_dtypes(include=['int64', "float64"]).columns.tolist()
    numerical_cols= X.select_dtypes(include=['object','category']).columns.tolist()

    if y in categorical_cols:
        categorical_cols.remove(y)
    if y in numerical_cols:
        numerical_cols.remove(y)
    return X,y, categorical_cols, numerical_cols


def visualize_dataset(X, y, categorical_cols, numerical_cols):
    print("Lets see which 2 columns (numerical) are very important and visualize our dataset using them")
    # using the ANOVA method to select the 2 best features
    X_num= X[numerical_cols]
    f_scores, p_values= f_classif(X_num, y)

    print("\n Feature F-scores: ")
    for i, score in enumerate(f_scores):
        print(f"Feature {i}: F-score = {score:.3}")

    best_features= np.argsort(f_scores)[-2:][::-1]
    print(f"\nBest 2 features: {best_features[0]} and {best_features[1]}")

    # Create DataFrame with best 2 features
    df_plot = pd.DataFrame({
        f'Feature_{best_features[0]}': X_num[:, best_features[0]],
        f'Feature_{best_features[1]}': X_num[:, best_features[1]],
        'Class': y
    })

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_plot,
        x=f'Feature_{best_features[0]}',
        y=f'Feature_{best_features[1]}',
        hue='Class',
        palette='Set2',  # or 'tab10', 'husl', 'rainbow'
        s=100,
        alpha=0.7,
        edgecolor='black'
    )
    plt.title('Class Separation: Best 2 Features', fontsize=14, fontweight='bold')
    plt.show()



def preprocess_data(X_train, X_test, categorical_cols, numerical_cols, method):
    """
    Preprocessing the data based on the selected method.
    """
    if method.lower()=="standardization":
        print("using standard scaler")
        if len(numerical_cols)>0:
            scaler= StandardScaler()    
            X_Scaled_train= X_train.copy()
            X_Scaled_test=X_test.copy()
            X_Scaled_train[numerical_cols]= scaler.fit_transform(X_train[numerical_cols])
            X_Scaled_test[numerical_cols]= scaler.transform(X_test[numerical_cols])

    if method.lower()=="normalization":
        print("using min-max-scaler")
        if len(numerical_cols)>0:
            scaler=MinMaxScaler()
            X_Scaled_train= X_train.copy()
            X_Scaled_test=X_test.copy()
            X_Scaled_train[numerical_cols]= scaler.fit_transform(X_train[numerical_cols])
            X_Scaled_test[numerical_cols]= scaler.transform(X_test[numerical_cols])

        # understand the categorical data in the dataset.
    if len(categorical_cols)>0:
        for col in categorical_cols():
            # using one-hot encoding
            ohe= OneHotEncoder(
                handle_unknown='ignore',
                drop='first',
            )
            # fit the encoder on data
            ohe.fit(X_train[categorical_cols])

            # transform and get the new columns names
            encoded_array= ohe.transform(X_train[categorical_cols])
            encoded_feature_names= ohe.get_feature_names_out(categorical_cols)

            # create the dataframe with encoded features
            encoded_df= pd.DataFrame(encoded_array, columns= encoded_feature_names, index=X_train.index)

            X_Scaled_train= X_train.drop(categorical_cols, axis=1)
            X_Scaled_train= pd.concat([X_Scaled_train, encoded_df], axis=1)


            # now for the test data
            encoded_test_array= ohe.transform(X_test[categorical_cols])
            encoded_feature_names_test=ohe.get_feature_names_out(categorical_cols)

            encoded_test_df= pd.DataFrame(encoded_test_array,columns= encoded_feature_names_test, index=X_test.index)

            X_Scaled_test= X_test.drop(categorical_cols, axis=1)
            X_Scaled_test= pd.concat([X_Scaled_test, encoded_test_df], axis=1)

    return X_Scaled_train, X_Scaled_test
            # using one hot encoding only since the user uploads arbritray data, how tf to know if there is a priority or not.
            # maybe mention or give user a chance to tell us?
            # lets seeeee, will figure out after the whole process
        

# note make some animated questions [answering 'how'] float around when the user presses Train machine

def train_test_splittng(X, y, train_size, test_size):
    # if size= float, [takes percentage]
    # if size= int, [takes that absolute number]
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=test_size,stratify=y, random_state=11)
    print(f"Total samples: {len(y)}")
    print(f"train samples: {len(y_train)}")
    print(f"test samples: {len(y_test)}")
    print("\nClass distribution (%)")
    for cls in np.unique(y):
        print(f"Class {cls}: "
              f"Train = {np.mean(y_train==cls)*100:.2f}% | "
              f"Test = {np.mean(y_test==cls)*100:.2f}%")
    return X_train, X_test, y_train, y_test


# we also have to do some visualizations related to trai test split
# i'll see that later

        
def select_model(model):
    print("yo, which algo you wanna run?")
    return model

# for the logistic regression using the One-vs-All approach
# building the class from scratch
# incase there are more than 2 classes to classify them into.
class BinaryLogistic:
    def __init__(self, regularization=None, epochs=1000, lr_=0.01, lambda_reg=0.01):
        self.m= None
        self.b= None
        self.losses=[]
        self.regularization= regularization
        self.lambda_reg= lambda_reg
        self.epochs=epochs
        self.lr_=lr_
    
    def sigmoid(self, z):
        return 1/(1+ np.exp(-z))
    
    # minimize the cross entropy loss
    def compute_losses(self, y_train, y_pred):
        """
        Binary cross-entropy loss:
        J = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
        """
        epsilon= 1e-15
        y_pred= np.clip(y_pred, epsilon, 1-epsilon)
        n= len(y_train)
        loss= -1/n * (y_train@ np.log(y_pred) + (1-y_train)@ np.log(1-y_pred))

        # adding regularization term (only on weights, not bias)
        if self.regularization=='l2':
            reg_term= self.lambda_reg * np.sum(self.m**2)
            loss += reg_term
        elif self.regularization=='l1':
            reg_term= self.lambda_reg * np.sum(np.abs(self.m))
            loss += reg_term

        return loss
    

    def fit(self, X_train,y_train, epochs):
        # use Xavier initialization. It is best for sigmoid
        n_features= X_train.shape[-1]
        self.m= np.random.randn(n_features) * np.sqrt(1/n_features)
        self.b= 0

        n= len(y_train)

        for epoch in range(self.epochs):
            linear_model= X_train @ self.m + self.b
            y_pred= self.sigmoid(linear_model)

            loss= self.compute_losses(y_train, y_pred)
            self.losses.append(loss)

            # compute gradients
            loss_slope_m= 1/n * X_train.T @ (y_pred - y_train)
            loss_slope_b= 1/n * np.sum(y_pred- y_train)

            # adding regularization gradients (only to weights)
            if self.regularization=='l2':
                loss_slope_m += 2 * self.lambda_reg * self.m
            elif self.regularization=='l1':
                loss_slope_m += self.lambda_reg * np.sign(self.m)

            # update weights
            self.m= self.m - self.lr_ * loss_slope_m
            self.b= self.b - self.lr_ * loss_slope_b

            if epoch % 100 == 0:
                print(f"Iteration {epoch}: Loss = {loss:.4f}")  

    def predict_proba(self, X):
        linear_model= X@self.m + self.b
        return self.sigmoid(linear_model)
    
    def predict(self, X):
        probabilities= self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    

class MulticlassLogistic:
    def __init__(self, lr=0.01, epochs=1000, regularization= None, lambda_=0.01):
        self.lr= lr
        self.epochs=epochs
        self.regularization= regularization
        self.lambda_= lambda_
        self.models= []
        self.classes= None

    def fit(self, X_train,y_train, verbose= False):
        self.classes= np.unique(y_train)
        n_classes= len(self.classes)

        if verbose:
            print(f"Training {n_classes} binary classifiers using One-vs-Rest...")
            print("=" * 60)

        # training a binary classifier for each class
        for idx, class_label in enumerate(self.classes):
            if verbose:
                print(f"\nClassifier {idx + 1}/{n_classes}: Class '{class_label}' vs Rest")

            # create binary labels: 1 if this class, otherwise 0
            binary_y= (y_train==class_label).astype(int)

            # create and train the binary classifier
            model= BinaryLogistic(
                lr_=self.lr,
                epochs= self.epochs,
                regularization=self.regularization,
                lambda_reg=self.lambda_
            )

            model.fit(X_train, binary_y, verbose=verbose)

            self.models.append(model)

        if verbose:
            print("="*60)
            print("Training done")

    
    def predicts_proba(self,X):
        probabilities= np.array([model.predict_proba(X) for model in self.models])
        class_indices= np.argmax(probabilities, axis=1)
        return self.classes[class_indices]
    

    def score(self, X,y):
        predictions= self.predict(X)
        return accuracy_score(y, predictions)
    



def DecisionTreesModel(X_train,X_test,y_train, y_test):
    dtree_classifier= DecisionTreeClassifier(random_state=11)
    dtree_classifier= dtree_classifier.fit(X_train, y_train)
    y_pred= dtree_classifier.predict(X_test)
    print("Confusion Matrix:\n")
    confusion_matrix(y_test, y_pred)
    print("Classification Report:\n")
    classification_report(y_test, y_pred)

def plot_tree(dtree_classifier):
    tree.plot_tree(dtree_classifier)









        


        


                

            

    


    


        







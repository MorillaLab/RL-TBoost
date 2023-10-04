#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from libraries import *
#from gtda.diagrams import PersistenceEntropy

classifiers = [LogisticRegression,KNeighborsClassifier,SVC,
               MLPClassifier,GaussianNB,DecisionTreeClassifier,
              RandomForestClassifier]


# In[1]:


def classification_results(x_dat,y_true,y_pred,mod,title=None,results_out= False):

#   print("Results for {}:".format(method.__name__))
    if title != None:
        print("Results for:",title)
    print(classification_report(y_true, y_pred))
    #print("Training Accuracy:",
          #round(classification_report(y_true, y_pred, output_dict=True)['accuracy'],4))
    print("Training Recall:", round(recall_score(y_true, y_pred, pos_label=0),4))
    print("Training Precision:",
          round(precision_score(y_true, y_pred, pos_label=0),4))
    print("Training F1-score:",
          round(f1_score(y_true, y_pred, pos_label=0),4))


    #print confusion matrix
    sns.set_palette("Paired")
    y_pred_rf = y_pred
    y_true_rf = y_true
    cm = confusion_matrix(y_true_rf, y_pred_rf)
    f, ax = plt.subplots(figsize =(5,5))
    sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="black",
                cmap="RdBu_r",fmt = ".0f",ax=ax)
    plt.xlabel("y_pred_rf")
    plt.ylabel("y_true_rf")
    plt.title('Training Data Confusion Matrix')
    plt.show()

    if results_out== True:
        return classification_report(y_true, y_pred, output_dict=True)


# In[2]:


def classification(method, x_dat, y_dat, model_out=False,
                   feature_importance=False, results=True,
                   test=False, resultOnly=False,
                   **params):

    #fit model
    mod = Pipeline([('classify', method(**params))])
    mod.fit(x_dat, y_dat)
    y_pred = mod.predict(x_dat)
    if (resultOnly == True) and (test == False):
        #print(classification_report(y_dat, y_pred, output_dict=True))
        return round(f1_score(y_dat, y_pred,pos_label=0),4)
   # round(classification_report(y_dat, y_pred, output_dict=True)['accuracy'],4)
    
    
    if (results == True) and (test == False):
        t = classification_results(x_dat,y_dat,y_pred,mod,title=method.__name__,results_out=results)

        if feature_importance == True:
            # Calculate permutation feature importance
            # (n_jobs=-1 means using all processors)
            try:
                imp = permutation_importance(mod, x_dat, y_dat, n_jobs=-1)

                #Generate feature importance plot
                plt.figure(figsize=(12,8))
                importance_data = pd.DataFrame({'feature':x_dat.columns, 'importance':imp.importances_mean})
                sns.barplot(x='importance', y='feature', data=importance_data)
                plt.title('Permutation Feature Importance')
                plt.xlabel('Mean Decrease in F1 Score')
                plt.ylabel('')
                plt.show()
                
                
            except:
                print('No Feature Importance Available')
        #return round(t['accuracy'],4)
        return round(f1_score(y_dat, y_pred, pos_label=0),4)


    if test != False:
        x_test, y_test = test[0], test[1]
        y_pred_test = mod.predict(x_test)
        
        if (resultOnly == True):
            #return round(classification_report(y_test, y_pred_test, output_dict=True)['accuracy'],4)
            return round(f1_score(y_test,y_pred_test, pos_label=0),4)
        else:
            print("Results for {}:".format(method.__name__))
            print(classification_report(y_test, y_pred_test))
            #print("Test Accuracy: {}%".format(round(mod.score(x_test, y_test)*100,2)))
            print("Testing Recall:", round(recall_score(y_test, y_pred_test, pos_label=0),4))
            print("Testing Precision:",
                  round(precision_score(y_test, y_pred_test, pos_label=0),4))
            print("Testing F1-score:",
                  round(f1_score(y_test, y_pred_test, pos_label=0),4))
            #print confusion matrix
            y_pred_rf = y_pred_test
            y_true_rf = y_test
            cm = confusion_matrix(y_test, y_pred_test)
            f, ax = plt.subplots(figsize =(5,5))
            sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
            plt.xlabel("y_pred_rf")
            plt.ylabel("y_true_rf")
            plt.title('Test Data Confusion Matrix')
            plt.show()
            #return round(classification_report(y_test, y_pred_test, output_dict=True)['accuracy'],4)
            return round(f1_score(y_test, y_pred_test, pos_label=0),4)


    if model_out == True:
        return mod
    


# In[3]:


def matrice_distance(data):
    d=euclidean_distances(data,data)
    d=preprocessing.normalize(d)
    return d


# In[4]:


def estimateur(data,d,sigma):
    f=[]
    for i in range(40):
        s=0
        for j in range(40):
            s=s+math.exp(- (d[i][j]**2)*(sigma**(-1)))
        f.append(s)
    return f


# In[5]:


def KL(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


# In[6]:
# Define a custom color mapping function
# Define a function to apply color mapping
def color_map(val):
    color = f'rgba({int(255 - val)}, 100, 100, 0.6)'
    return f'background-color: {color};'


def showResult(classifiers,x_train,y_train,test=False,resultOnly=False):
    
    result = []
    classifiers_columns = []
    for cls in classifiers:
        print('_' * 50)
        print('-' * 50)
        classifiers_columns.append(cls.__name__)
        ans = classification(cls,x_train,y_train,test=test)
        result.append(ans)
    result = pd.DataFrame([result], columns=classifiers_columns)
    # Apply the color mapping function to the DataFrame
    styled_df = result.style.applymap(color_map)

    # Display the styled DataFrame
    return styled_df


# In[7]:


def topologicalFeatures(classifiers,x,y,test=False,results=True,resultOnly=True,model_out=True,
                        random_state = 7):
    x = x.to_numpy().reshape(x.shape[0],x.shape[1],1)
    homology_dimensions = [0, 1, 2]
    result = []
    column_labels = ["bottleneck", "wasserstein", "landscape", "betti", "heat", "silhouette", "persistence_image"]
    
    for metric in column_labels:
        steps = [
            ("persistence", VietorisRipsPersistence(metric="euclidean", homology_dimensions=homology_dimensions, n_jobs=6)),
            ("amplitude",Amplitude(metric, n_jobs=-1)),
        ]

        pipeline = Pipeline(steps)
        data = pipeline.fit_transform(x)
        res = []
        classifiers_columns = []
        if test!=False:
            test_data = test[0].to_numpy().reshape(test[0].shape[0],test[0].shape[1],1)
            test_data = pipeline.fit_transform(test_data)
            for cls in classifiers:
                classifiers_columns.append(cls.__name__)
                print("*"*40)
                res.append(classification(cls,data,y,test=[test_data,test[1]],results=result,resultOnly=result,model_out=model_out))
                print("+"*40)
        else:
            for cls in classifiers:
                classifiers_columns.append(cls.__name__)
                res.append(classification(cls,data,y,results=result,resultOnly=result,model_out=model_out,random_state=random_state))
        result.append(res)
        
    column_labels.append("entropy")
    steps = [
        ("persistence", VietorisRipsPersistence(metric="euclidean", homology_dimensions=homology_dimensions, n_jobs=6)),
        ("entropy",PersistenceEntropy(normalize=True)),
    ]
    pipeline = Pipeline(steps)
    res = []
    data = pipeline.fit_transform(x)
    
    
    if test!=False:
        test_data = test[0].to_numpy().reshape(test[0].shape[0],test[0].shape[1],1)
        test_data = pipeline.fit_transform(test_data)
        classifiers_columns = []
        for cls in classifiers:
            classifiers_columns.append(cls.__name__)
            res.append(classification(cls,data,y,test=[test_data,test[1]],results=result,resultOnly=result,model_out=model_out))
    else:
        for cls in classifiers:
            classifiers_columns.append(cls.__name__)
            res.append(classification(cls,data,y,results=result,resultOnly=result,model_out=model_out,random_state=random_state))
    result.append(res)
    
    
    return pd.DataFrame(result, index=column_labels,columns=classifiers_columns)


# In[8]:


def topologicalFeaturesComplete(x,y,test=False):
    from gtda.diagrams import PersistenceEntropy
    from gtda.homology import VietorisRipsPersistence
    x = x.to_numpy().reshape(x.shape[0],x.shape[1],1)
    homology_dimensions = [0, 1, 2]
    result = []
    column_labels = ["bottleneck", "wasserstein", "landscape", "betti", "heat", "silhouette", "persistence_image"]
    classifiers_columns = ["LogisticRegression","KNeighborsClassifier","SVC",
               "MLPClassifier","GaussianNB","DecisionTreeClassifier",
              "RandomForestClassifier"]
    for metric in column_labels:
        steps = [
            ("persistence", VietorisRipsPersistence(metric="euclidean", homology_dimensions=homology_dimensions, n_jobs=6)),
            ("amplitude",Amplitude(metric, n_jobs=-1)),
        ]

        pipeline = Pipeline(steps)
        data = pipeline.fit_transform(x)
        res = []
        if test!=False:
            test_data = test[0].to_numpy().reshape(test[0].shape[0],test[0].shape[1],1)
            test_data = pipeline.fit_transform(test_data)
            for cls in classifiers:
                res.append(classification(cls,data,y,test=[test_data,test[1]],results=True,resultOnly=True))
        else:
            for cls in classifiers:
                res.append(classification(cls,data,y,results=True,resultOnly=True))
        result.append(res)
        
    column_labels.append("entropy")
    steps = [
        ("persistence", VietorisRipsPersistence(metric="euclidean", homology_dimensions=homology_dimensions, n_jobs=6)),
        ("entropy",PersistenceEntropy(normalize=True)),
    ]
    pipeline = Pipeline(steps)
    res = []
    data = pipeline.fit_transform(x)
    
    
    if test!=False:
        test_data = test[0].to_numpy().reshape(test[0].shape[0],test[0].shape[1],1)
        test_data = pipeline.fit_transform(test_data)
        for cls in classifiers:
            res.append(classification(cls,data,y,test=[test_data,test[1]],results=True,resultOnly=True))
    else:
        for cls in classifiers:
            res.append(classification(cls,data,y,results=True,resultOnly=True))
    result.append(res)
    
    
    return pd.DataFrame(result, index=column_labels,columns=classifiers_columns)

def tda_tranform(metrics,data):
    #daa must be of type numpy
    # Creating the diagram generation pipeline
    diagram_steps = [
       VietorisRipsPersistence(homology_dimensions=[0, 1, 2]),
    ]

    # Select a variety of metrics to calculate amplitudes
    metric_list = [
        {"metric": metric}
        for metric in metrics
    ]

    # Concatenate to generate 3 + 3 + (4 x 3) = 18 topological features
    feature_union = make_union(
        *[PersistenceEntropy(nan_fill_value=-1)]
        + [Amplitude(**metric, n_jobs=-1) for metric in metric_list]
    )

    # Create the diagram generation pipelines
    pipeline_list = [
        make_pipeline(diagram_step, feature_union)
        for diagram_step in diagram_steps
    ]

    # Create the final TDA union pipeline
    tda_union = make_union(*pipeline_list, n_jobs=-1)
    data = data.reshape(data.shape[0],data.shape[1],1)
    return tda_union.fit_transform(data)
    
def modelTunning(params,x,y,estimator,scoring):
    gscv = GridSearchCV(estimator=estimator,param_grid=params,
                                 n_jobs=3, scoring = scoring)
    gscv.fit(x,y)

    print(gscv.best_estimator_)
    print(gscv.best_params_)
    print("Best Score:", gscv.best_score_)
    
    y_pred= gscv.predict(x)
    method_name = type(estimator).__name__
    print("Method")
    print(method_name)
    print(f"Results for Accuracy Optimized {method_name} GSCV")
    classification_results(x,y,y_pred,gscv)
    return [gscv,y_pred]

def best_threshold(model,x,y):
    #getting the best threshold for prediction in test dataset
    thresh = [0.2,0.25,0.3,0.35,0.4,0.45,0.5]

    precision=[]
    recall=[]

    # Using the LogReg GSCV Accuracy Optimized Model
    for t in thresh:
        print('_' * 50)
        print('-' * 50)
        y_pred_thresh = (model.predict_proba(x)[:,1] >= t).astype(bool)
        classification_results(x,y,y_pred_thresh,model, title=t)

        recall.append(round(recall_score(y, y_pred_thresh, pos_label=0),4))
        precision.append(round(precision_score(y, y_pred_thresh, pos_label=0),4))

    plt.plot(thresh,recall, label="Recall")
    plt.plot(thresh,precision, label = 'Precision')
    plt.title("Threshold of Recall vs Precision")
    plt.legend()
    plt.show()

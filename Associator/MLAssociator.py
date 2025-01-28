from Associator.basic_associator import BasicAssociator
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import track as tk
import constants 
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import Plot.plotter as plot

import math
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import datetime

from tensorflow.keras.losses import MeanSquaredLogarithmicError

#import tensorflow_decision_forests as tfdf
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_val_score
from sklearn import metrics


class MLAssociator(BasicAssociator):

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.name = "ML_ASSOCIATOR"

    def associate(self, data, train_data=None):

        self.association_results.tic()
        if "train" in self.config:            
            print("TRAINING MODEL")
            print("----------------------")
            self.ml_train(train_data, self.config["train"])
        # print(data)
        super().code_association(data)
        self.association_results.baselineAssociations()
        print("Number of tracks: " +str(len(self.association_results.tracks)))
        self.ml_association(data)
        self.association_results.end_association()
        

        self.association_results.toc()

        return self.association_results
        
    #@profile
    def ml_association(self, data):
        ## Load model from config          
        print(data)
        self.model = keras.models.load_model(self.config["model"])
        total_length = float(len(data))
        #TODO: change from config
        scaler = pickle.load(open(self.config["scaler"], 'rb'))
        AssociationChecks = 0
        Associations = 0
        dropped_order = 0
        dropped_time = 0 
        dropped_pos = 0
        dropped_time = 0
        dropped_inactive = 0
        count = 0
        last = 0
        inactive_tracks = []

        for i, point in data.iterrows():
            check = 0
            count +=1
            if int(count/total_length * 100) > last:
                print("Progress " + str((count/total_length * 100)))
                print("time" + str(datetime.datetime.now().time()))
                print("Number of associations " + str(Associations))
                print("Number of checks " + str(AssociationChecks))
                print("Number of dropped_order " + str(dropped_order))
                print("Number of dropped_time " + str(dropped_time))
                print("Number of dropped_pos " + str(dropped_pos))
                print("Number of dropped_inactive " + str(dropped_inactive))
                last =  int(count/total_length * 100)
                
            best_value = 0.5
            best_track = None

            X = []
            candidates = []

            #Clear inactive tracks
            if inactive_tracks != []:
                #print(inactive_tracks)
                #print(self.association_results.tracks)
                self.association_results.tracks = [i for i in self.association_results.tracks if i not in inactive_tracks]
            inactive_tracks = []
            for track in self.association_results.tracks:
                #tod diff and pos diff
                if not track.active:
                    dropped_inactive += 1
                    continue

                #print(t)
                #print(t["TOD"])
                #print(point)
                #print(point["TOD"])

                if point["TOD"] < track.min_T:
                    dropped_order += 1
                    continue
                if point["TOD"] - track.max_T > 60:
                    dropped_time += 1
                    track.set_inactive()
                    
                    continue


                
                tail = find_prev_points(track.points[["X", "Y", "Z","TOD", "TARGET_ADDR"]], point["TOD"])
               # print(tail)
                tail = tail - point[["X", "Y", "Z","TOD", "TARGET_ADDR"]]
               # print(tail)
                if(len(tail) < 2):
                    continue

                last_ = tail.tail(1)
                if (last_["TOD"] < - 60).any():
                    dropped_time += 1
                    continue
                    
                if (np.sqrt(last_.iloc[0]['X']**2 + last_.iloc[0]['Y']**2) > 15000).any():
                     dropped_pos += 1
                     continue
                check += 1
                    
                if(tail["TOD"].max() > 0):
                    print("Something phisy")
                    print(point)
                    print(track.points)
                    print("time to check: ")
                    print(point["TOD"])
                    print(track.points["TOD"] <= point["TOD"])
                    print(track.points[track.points["TOD"] <= point["TOD"]]["TOD"])
                    maxindx = track.points[track.points["TOD"] <= point["TOD"]]["TOD"].idxmax()
                    print(maxindx)
                    print(track.points[track.points["TOD"] <= point["TOD"]].tail(10))
                    print(track.points[:maxindx].tail(11))

                    stop
                with pd.option_context('future.no_silent_downcasting', True):
                    x = expandTraj(tail, 9).tail(1).fillna(0) 
               
                #print(x)
                
                scaled = scaler.transform(x)

                X.append(scaled)
                candidates.append(track)

            
            AssociationChecks += check
            check = 0
            if len(X) > 0:
                #print(X)
                result = self.model.predict(np.concatenate( X, axis=0 ), verbose=0)
                index = np.argmax(result)
                if result[index] > best_value:
                    Associations += 1
                    candidates[index].add(point, phase="ML_PREDICTION")


                
                
        print("Number of associations " + str(Associations))
        print("Number of checks " + str(AssociationChecks))





    def ml_train(self, data, config):
        print("Training for "+ str(len(data)) + " items.")
        X = pd.DataFrame()
        Y = pd.DataFrame()
        if config["generate_data"]:
            for taddr in tk.cleanCodeList(data["TARGET_ADDR"].unique()):
                rows = data.loc[data['TARGET_ADDR'] == taddr].copy()
                #data.drop(rows.index, inplace = True)
                rows.reset_index(drop = True, inplace = True)
                rows["time_diffs"] = rows["TOD"].diff()
                with pd.option_context('future.no_silent_downcasting', True):
                    rows["time_diffs"].fillna(0)
                split_indices = rows[rows["time_diffs"] >= constants.MAX_T_DIFF_MODEA]
                split_indices = [0] + split_indices.index.tolist() + [len(rows) - 1]
                # split the DataFrame based on the new grouping variable
                dfs = [rows.iloc[split_indices[i]:split_indices[i + 1]].drop(columns=['time_diffs']) for i in range(len(split_indices) - 1)]
            
                for row in dfs:
                    tl , tly = self.generateDataset(row[["X", "Y", "Z", "TOD", "TARGET_ADDR"]], taddr, data[["X", "Y", "Z", "TOD", "TARGET_ADDR"]])
                    
                    #print(tl)
                    #print(tly)
                    X = pd.concat([X, pd.DataFrame(tl)],  ignore_index=True)
                    Y = pd.concat([Y, pd.DataFrame(tly)], ignore_index=True)
                    
                    
                    inds = pd.isnull(X).any(axis=1).to_numpy().nonzero()[0]
                    
                    X.drop(inds, inplace = True)
                    Y.drop(inds, inplace = True)
                
            
            
            X = center_on_m(X)
            # print(X)
            X = X.drop(columns= ["X_m", "Y_m", "Z_m", "TOD_m"])
            # print(X)
            # print(Y)
            X.to_pickle(config["X_dataset"])
            Y.to_pickle(config["Y_dataset"])
            
            
        else:
            X = pd.read_pickle(config["X_dataset"])
            Y = pd.read_pickle(config["Y_dataset"])
        if config["SCALER_MODE"] == "robust":
            scaler = RobustScaler()
        if config["SCALER_MODE"] == "scaler":
            scaler = StandardScaler()

        
        
        x_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        with pd.option_context('future.no_silent_downcasting', True):
            x_scaled = x_scaled.fillna(0)
        #print(x_scaled)
        
        ros = RandomOverSampler(
            sampling_strategy='auto',
            random_state=0,
            shrinkage = 1
        )

        #X_res, y_res = ros.fit_resample(x_scaled, Y)
        X_res = x_scaled
        y_res = Y

        y_res.to_pickle("y_res.pkl")
        X_res.to_pickle("x_res.pkl")

        
        
        pickle.dump(scaler, open(self.config["scaler"], 'wb'))
        # kFold = StratifiedKFold(n_splits=5)
        # print(X_res)
        # for i, (train_index, test_index) in enumerate(kFold.split(X_res, y_res)):
        
        #     print(f"Fold {i}:")

        #     #XTraining, XValidation, YTraining, YValidation = train_test_split(X_res.values, y_res.astype(float).values, stratify= y_res.astype(float).values, test_size=0.2) # before model building
            
        #     model = build_model_using_sequential(10)

        #     stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        
        #     history = model.fit(X_res.values[train_index],y_res.astype(float).values[train_index],batch_size=256,epochs=150,
        #                         validation_data=(X_res.values[test_index],y_res.values.astype(float)[test_index]), callbacks = [stop_early])
        #     print("History") 
        #     print(history)
            
        #     print(f"\n\n\nEvaluation Fold {i}:")
        #     print(model.evaluate(X_res.values[test_index],y_res.astype(float).values[test_index]))
        #     print("\n\n\n\n\n")
    
        # model.save(self.config["model"])
        XTraining, XValidation, YTraining, YValidation = train_test_split(X_res.values, y_res.astype(float).values, stratify= y_res.astype(float).values, test_size=0.2) # before model building

        
        model = build_model_using_sequential(10)

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
        history = model.fit(XTraining,YTraining,batch_size=256,epochs=150,
                            validation_data=(XValidation,YValidation), callbacks = [stop_early])

        model.save(self.config["model"])
        
        

    def generateDataset(self, traj, taddr, TRs):
        #N = constants.NUMBER_TRS_MODEL
        N = 9

        exp_traj = expandTraj(traj, N)

        l = []
        ly = []
        count = 0
    
        for idx,row in exp_traj.iterrows():
            subTR = self.filter(TRs, row)

            for idx, item in subTR.iterrows():

                temp = item.drop(["TARGET_ADDR"])

                temp.index = [f'{col}_m' for col in temp.index]

                l.append(pd.concat([row, temp], axis = 0 ))
                ly.append(int(item["TARGET_ADDR"]) == taddr)
                del temp
            del subTR
    
        return l, ly
        
    def filter(self, TRs, rows):

        filtered = TRs.copy()
        filtered["Tdiff"] = filtered["TOD"] - rows["TOD"]
        filtered = filtered[(filtered['Tdiff'] > 0) & (filtered['Tdiff'] <= self.config["train"]["MAX_TDIFF"])]
    
        filtered["Xdiff"] = filtered["X"] - rows["X"]
        filtered["Ydiff"] = filtered["Y"] - rows["Y"]
    
        filtered["dist"] = (filtered["Xdiff"]**2 + filtered["Ydiff"]**2)**(1/2)
        filtered = filtered[filtered['dist'] <= self.config["train"]["MAX_POSDIFF"]]
    
        return filtered.drop(["dist", "Xdiff", "Ydiff", "Tdiff"], axis= 1)


def find_prev_points(traj, TOD):

    mask = traj["TOD"] <= TOD  
    return traj[mask].tail(10)

def expandTraj(traj, N):
    result_df = traj.drop(['TARGET_ADDR'], axis=1)
    shifted_df = result_df
    cols = shifted_df.columns
    for i in range(N):
        shifted_df = shifted_df.shift(1)
        shifted_df.columns = [f'{col}_{i+1}' for col in cols]
        result_df = pd.concat([shifted_df, result_df], axis=1)
    return result_df

def center_on_m(X):

    X["X_9"] = X["X_9"] - X["X_m"]
    X["X_8"] = X["X_8"] - X["X_m"]
    X["X_7"] = X["X_7"] - X["X_m"]
    X["X_6"] = X["X_6"] - X["X_m"]
    X["X_5"] = X["X_5"] - X["X_m"]
    X["X_4"] = X["X_4"] - X["X_m"]
    X["X_3"] = X["X_3"] - X["X_m"]
    X["X_2"] = X["X_2"] - X["X_m"]
    X["X_1"] = X["X_1"] - X["X_m"]
    X["X"] = X["X"] - X["X_m"]

    X["Y_9"] = X["Y_9"] - X["Y_m"]
    X["Y_8"] = X["Y_8"] - X["Y_m"]
    X["Y_7"] = X["Y_7"] - X["Y_m"]
    X["Y_6"] = X["Y_6"] - X["Y_m"]
    X["Y_5"] = X["Y_5"] - X["Y_m"]
    X["Y_4"] = X["Y_4"] - X["Y_m"]
    X["Y_3"] = X["Y_3"] - X["Y_m"]
    X["Y_2"] = X["Y_2"] - X["Y_m"]
    X["Y_1"] = X["Y_1"] - X["Y_m"]
    X["Y"] = X["Y"] - X["Y_m"]

    X["Z_9"] = X["Z_9"] - X["Z_m"]
    X["Z_8"] = X["Z_8"] - X["Z_m"]
    X["Z_7"] = X["Z_7"] - X["Z_m"]
    X["Z_6"] = X["Z_6"] - X["Z_m"]
    X["Z_5"] = X["Z_5"] - X["Z_m"]
    X["Z_4"] = X["Z_4"] - X["Z_m"]
    X["Z_3"] = X["Z_3"] - X["Z_m"]
    X["Z_2"] = X["Z_2"] - X["Z_m"]
    X["Z_1"] = X["Z_1"] - X["Z_m"]
    X["Z"] = X["Z"] - X["Z_m"]

    X["TOD_9"] = X["TOD_9"] - X["TOD_m"]
    X["TOD_8"] = X["TOD_8"] - X["TOD_m"]
    X["TOD_7"] = X["TOD_7"] - X["TOD_m"]
    X["TOD_6"] = X["TOD_6"] - X["TOD_m"]
    X["TOD_5"] = X["TOD_5"] - X["TOD_m"]
    X["TOD_4"] = X["TOD_4"] - X["TOD_m"]
    X["TOD_3"] = X["TOD_3"] - X["TOD_m"]
    X["TOD_2"] = X["TOD_2"] - X["TOD_m"]
    X["TOD_1"] = X["TOD_1"] - X["TOD_m"]
    X["TOD"] = X["TOD"] - X["TOD_m"]


    return X


def build_model_using_sequential(hp):
  hidden_units1 = 512
  hidden_units2 = 512
  hidden_units3 = hidden_units2
  hidden_units4 = 512
  model = Sequential()
  model.add(Dense(hidden_units1, kernel_initializer='normal', activation='relu', input_dim=40))
  model.add(Dropout(0.2))
  for l in range(hp-1):
      model.add(Dense(hidden_units1, kernel_initializer='normal', activation='relu'))
      model.add(Dropout(0.2))

  model.add(Dense(hidden_units1, kernel_initializer='normal', activation='relu'))
  model.add( Dense(1, kernel_initializer='normal', activation='sigmoid'))
  learning_rate = 1e-4
  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate) ,
    loss='binary_crossentropy',
    metrics=['binary_accuracy', tf.keras.metrics.Precision(name="precision"),
              tf.keras.metrics.Recall(name="recall"),
              tf.keras.metrics.F1Score(name="f1", threshold=0.5)]
  )
  return model
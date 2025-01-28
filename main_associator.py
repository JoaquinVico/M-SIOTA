import sys
from datetime import datetime
import json
import pandas as pd
import numpy as np
import pymap3d as pm
import track as tk

from DataReader import openSky_loader as opsk
from Associator.basic_associator import BasicAssociator
from Associator.MLAssociator import MLAssociator
from sklearn.model_selection import train_test_split

class AssociationManager:
    '''
    Configurable parameters:
    - Database read:
        2 available data sources (OpenSky csv and database)
        for database IP and source type are available
        for OpenSky, filepaht is required

    - Data formatting configuration
        hidden modeS is available [0-1]: % of trs that will have their ModeS code removed.
    - Associator (list):
        - Class type (ML)
        - Configuration specific to type (train/load...)
    - Evaluator:
        - Metrics to obtain and display/storage options
    '''

    def __init__(self, config_route=None):
        if config_route is None:

            ##Create default config file
            default = {}

            ## Default DB
            #default["db"] = {"ip": "", "source": 1}

            ## Default OpenSky
            default["OpenSky"] = {"filename": "./opensky/states_2022-06-27-23.csv"}

            # Transformations
            default["transformations"] = {"hide_modeS": 0.4, 
                                          #"number_of_targets":10,
                                          "filter_pos":{ #for OPSK
                                              "lat": 49.58558260032726,
                                              "lon":  9.590482986840815,
                                              "thr": 20
                                          },
                                          #"filter_pos":{
                                          #    "lat":49.52191115226883,
                                          #    "lon":  6.23444436761613,
                                          #    "thr": 2.5
                                          #},
                                          
                                          
                                          #"filter_time":[1656300790, 1656374390], #Filter for opsk
                                          "filter_pos_BCN":{
                                              "lat": 41.290652474073184,
                                              "lon": 2.0801125019352416,
                                              "thr": 2.5
                                          },
                                          "drop_no_id":True,
                                          "pos_only": True,
                                          #"pos_only_latlon":True,
                                          #"reduce":250000
                                          #"drop_nopos_update": 3
                                         } 
                                          

            ## Default Associator
            # associator modes
            trainning_ML = {"class": "ml", 
                            "train": {
                                "generate_data": True,
                                "X_dataset": "X_new.pkl",
                                "Y_dataset": "Y_new.pkl",
                                "MAX_TDIFF": 30,
                                "MAX_POSDIFF": 10000,
                                #"SCALER_MODE": "robust",
                                "SCALER_MODE": "scaler",
                                "split": 0.8 #training split
                            },
                            "model":"./Data/OPSK_Trained.keras",
                            "scaler": "./OPSK_robust_scaler.pkl"
                           }
            eval_ML = {"class": "ml", 
                            "model":"./Data/OPSK_Trained.keras",
                            "scaler": "./OPSK_robust_scaler.pkl"
                           }     

            
            
            
            default["associator"] = [trainning_ML]
            ## Default Evaluator
            path = "evaluation_" + str(datetime.today().strftime('%Y-%m-%d'))
            default["evaluator"] = {"output": ["tracks", "csv", "cmd"], "path": path}

            default["debug"] = {"data": True}
            self.configuration = default

            # save as default
            with open('default_config.json', 'w', encoding='utf-8') as f:
                json.dump(default, f, ensure_ascii=False, indent=4)
        else:
            print("Loading config " + str(config_route))
            with open(config_route, 'r', encoding='utf-8') as f:
                self.configuration = json.load(f)


    ## load the configuration for the Association Manager
    def readData(self):

        if "db" in self.configuration:
            database = data_retriever.createDatabase(ip=self.configuration["db"]["ip"], source=self.configuration["db"]["source"])
            if self.configuration["db"]["source"] == 1:
                data = database.get_all_data("sensorData", database.adsb["DS_ID"].values[0])#Todo: change
            else:
                data = database.get_all_data("sd_ads", database.adsb["DS_ID"].values[0])#Todo: change
            print("ADS read entries", len(data))

            for radar in database.radar["DS_ID"].values:
                print("Processing: " + str(radar))
                if self.configuration["db"]["source"] == 1:
                    data = pd.concat([data, database.get_all_data("sensorData", radar)], ignore_index=True)
                else:
                    data = pd.concat([data, database.get_all_data("sd_radar", radar)], ignore_index=True)
                
                print("Radar  read", len(data))

            print("Database read finished")

            if self.configuration["debug"]["data"]:
                print("Histogram plot of detections by TOD")
                data.hist("TOD")
                print("Unique TADDRs")
                print(data["DS_ID"].unique())
        elif "OpenSky" in self.configuration:
            data = opsk.process_csv(self.configuration["OpenSky"]["filename"])
            if "filter_pos" in self.configuration["transformations"]:
                droplist = (abs(data["POS_LAT_DEG"] - self.configuration["transformations"]["filter_pos"]["lat"]) > self.configuration["transformations"]["filter_pos"]["thr"]) | (abs(data["POS_LONG_DEG"] - self.configuration["transformations"]["filter_pos"]["lon"]) > self.configuration["transformations"]["filter_pos"]["thr"] )
                data.drop(data[droplist].index, inplace=True)   
            data["X"], data["Y"], data["Z"] = pm.geodetic2enu(data["POS_LAT_DEG"], data["POS_LONG_DEG"], data["H_latlon"], data["POS_LAT_DEG"].mean(), data["POS_LONG_DEG"].mean(), 0)
            data["Z"] = data["H_latlon"]

        return data

    def formatData(self, data):

        ## Data formating
        # From the database read, the data is already formatted correctly, here "fixes" are done with the data
        # specifically for the current analisys
        # E.G.: mark a certain percentaje of the TRs as fake PSR for latter analysis
        #data.dropna(axis=1, how="all", inplace=True)
        if "pos_only" in self.configuration["transformations"] and self.configuration["transformations"]["pos_only"]:
            data = data.loc[:,["X","Y","Z", "TOD","TARGET_ADDR"]]

        if "pos_only_latlon" in self.configuration["transformations"] and self.configuration["transformations"]["pos_only_latlon"]:
            data = data.loc[:,["POS_LAT_DEG", "POS_LONG_DEG", "X","Y","Z", "TOD","TARGET_ADDR"]]
            
        data["original_Taddr"] = data["TARGET_ADDR"]

        # Only for OpenSky data, if no pos has been received in some time, drop the TR
        if "drop_nopos_update" in self.configuration["transformations"] and "lastposupdate" in data.columns:
            droplist = data["TOD"] - data["lastposupdate"] > self.configuration["transformations"]["drop_nopos_update"]
            data.drop(data[droplist].index, inplace=True)
            
        if "drop_no_id" in self.configuration["transformations"] and self.configuration["transformations"]["drop_no_id"]:
            taddress_list = tk.cleanCodeList(data["TARGET_ADDR"].unique())
            data = data[data['TARGET_ADDR'].isin(taddress_list)]

        if "number_of_targets" in self.configuration["transformations"]:
            taddress_list = tk.cleanCodeList(data["TARGET_ADDR"].unique())
            taddress_list = taddress_list[0:self.configuration["transformations"]["number_of_targets"]]
            print(taddress_list)
            data = data[data['TARGET_ADDR'].isin(taddress_list)]
            #print(data["TARGET_ADDR"].unique())

        if "hide_modeS" in self.configuration["transformations"]:
            data["hidden_modeS"] = np.random.rand(data.shape[0]) < self.configuration["transformations"]["hide_modeS"]
            data["TARGET_ADDR"] = data.apply(lambda row: int(0) if row['hidden_modeS'] else int(row['TARGET_ADDR']), axis=1)
        else:
            data["hidden_modeS"] = False

        if "reduce" in self.configuration["transformations"]:
            data = data.head(self.configuration["transformations"]["reduce"])

        

        return data

    def associate(self, data):
        association_results = []
        num =  len(self.configuration["associator"])
        count = 1
        unique_ids = data["original_Taddr"].unique()
        train_flights, test_flights = train_test_split(unique_ids, test_size=0.2) 

        train_data = data[data['original_Taddr'].isin(train_flights)]
        test_data = data[data['original_Taddr'].isin(test_flights)]

        # #for TMA 
        # train_data, test_data = train_test_split(data, test_size=0.2) 

        print("Test data")
        print(test_data)

        print("\n\n\n Train data")
        print(train_data)

        for associator in self.configuration["associator"]:
            type = associator["class"]
            print("Associating with " + type + " num " + str(count) + " of " + str(num))
            if type == "basic":
                asoc = BasicAssociator()
            if type == "ml":
                asoc = MLAssociator(associator)

            results = asoc.associate(test_data.copy(), train_data.copy())
            results.end_association()
            association_results.append(results)

        print("Association finished")

        return association_results


    def evaluate(self, data, association_results):
        evaluation = {}
        for i, result in enumerate(association_results):

            ## Truth metrics:
            num_associations = result.count_TRs()
            num_losses =  result.total_reports- num_associations
            ratio_losses = num_losses / result.total_reports

            correct_associations = result.correctAssociations()
            false_associations = result.falseAssociations()

            evaluation[i] = {"total_detections": result.total_reports, "total_associations": num_associations, "misses": num_losses,
                             "correct_associations": correct_associations, "mismatches": false_associations}


        #print
        for eval in evaluation:
            print(eval)
        print(evaluation)
        #export to file
        #plot



if __name__ == "__main__":

    if len(sys.argv) > 1:
        manager = AssociationManager(sys.argv[1])
    else:
        manager = AssociationManager()
    print(manager.configuration)

    data = manager.readData()

    print(data)

    data = manager.formatData(data)
    print("Modified Data \n")
    print(data)

    association_results = manager.associate(data)

    print(association_results)

    manager.evaluate(data, association_results)


#%%

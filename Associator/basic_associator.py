import datetime
import constants
import track as tk
import numpy as np
import pandas as pd
import pickle


class track_result():
    def __init__(self, id, points, taddr=""):
        self.id = id
        self.taddr = taddr
        self.points = points
        self.points["PHASE"] = "basic"
        self.active = True
        self.min_T = self.points["TOD"].min()
        self.max_T = self.points["TOD"].max()

    def add(self, point, phase="basic"):
        
        if self.taddr == "" and not np.isnan(point["TARGET_ADDR"]) and point["TARGET_ADDR"] != 0:
            self.taddr = point["TARGET_ADDR"]
        point["PHASE"] = phase
        self.points.loc[point.name] = point.values
        self.points.sort_values(by=["TOD"], inplace=True)
        self.min_T = self.points["TOD"].min()
        self.max_T = self.points["TOD"].max()

    def set_inactive(self):
        self.active = False
        #pickle.dump(self, open("tracks/"+str(self.id) +".pkl", 'wb'))
        


class AssociationResults:
    ## Association results must contain:
    #   Tracks with ID, points, ??
    #   associator name
    #   association time
    def __init__(self):
        self.total_reports = 0
        self.itime = 0
        self.endtime = 0
        self.run_time = 0
        self.tracks = []
        self.inactive_tracks = []
        self.baseline = 0
        self.name = "BasicAssociator"

    def tic(self):
        self.itime = datetime.datetime.now()

    def toc(self):
        self.endtime = datetime.datetime.now()
        self.run_time = self.endtime - self.itime

    def add_track(self, points, taddr=""):
        self.tracks.append(track_result(len(self.tracks), points, taddr=taddr))
        
    def end_association(self):
        for track in self.tracks:
            track.set_inactive()
            self.inactive_tracks.append(track)
        self.tracks = []

    def count_TRs(self):
        count = 0
        for track in self.inactive_tracks:
            count += len(track.points)

        return count

    def correctAssociations(self):
        count = 0
        for track in self.inactive_tracks:
            if track.taddr != "":
                # pandas dataframe: track.points
                count += len(track.points[track.points["original_Taddr"] == float(track.taddr)])
            # Review non taddr available
        # print(count)
        return count

    def falseAssociations(self):
        count = 0
        for track in self.inactive_tracks:
            if track.taddr != "":
                # pandas dataframe: track.points
                count += len(track.points[track.points["original_Taddr"] != float(track.taddr)])
            # Review non taddr available
        return count
        
    def unknownAssociations(self):
        count = 0
        for track in self.inactive_tracks:
            if track.taddr != "":
                # pandas dataframe: track.points
                count += len(track.points[track.points["original_Taddr"] == 0.0])
            # Review non taddr available
        return count

    def baselineAssociations(self):
        count = 0
        for track in self.tracks:
            if track.taddr != "":
                # pandas dataframe: track.points
                count += len(track.points[track.points["original_Taddr"] == float(track.taddr)])
            # Review non taddr available
        self.baseline = count

    def missedTracks(self):
        return 0
    def duplicatedTracks(self):
        count = 0
        for track in self.tracks:
            taddr = float(track.taddr)
            for track2 in self.tracks:
                if float(track2.taddr) == taddr:
                    if track.points["TOD"].min() <= track2.points["TOD"].max() and track.points["TOD"].max() >= track2.points["TOD"].min():
                        count += 1
        return count
        
        

class BasicAssociator():

    def __init__(self):
        self.association_results = AssociationResults()

    def associate(self, data, train_data = None):
        self.association_results.tic()
        self.code_association(data)
        self.association_results.baselineAssociations()
        self.association_results.toc()

        return self.association_results

    def code_association(self, data):
        self.association_results.total_reports = len(data)
        print("Aircrafts Identified " + str(len(tk.cleanCodeList(data["TARGET_ADDR"].unique()))))
        for taddr in tk.cleanCodeList(data["TARGET_ADDR"].unique()):
            rows = data.loc[data['TARGET_ADDR'] == taddr].copy()
            data.drop(rows.index, inplace = True)
            rows.reset_index(drop = True, inplace = True)
            rows["time_diffs"] = rows["TOD"].diff()
            rows["time_diffs"].fillna(0)
            split_indices = rows[rows["time_diffs"] >= constants.MAX_T_DIFF_MODEA]
            split_indices = [0] + split_indices.index.tolist() + [len(rows) - 1]
            # split the DataFrame based on the new grouping variable
            dfs = [rows.iloc[split_indices[i]:split_indices[i + 1]].drop(columns=['time_diffs']) for i in range(len(split_indices) - 1)]
            for row in dfs:

                if len(row) > constants.MIN_TRACK:
                    self.association_results.add_track(row, taddr=taddr)




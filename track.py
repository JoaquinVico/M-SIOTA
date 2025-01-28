import numpy as np
import pandas as pd


def calculateCovarianceMatrix(track):
    try:
        df = track[["X", "Y"]].diff(periods=1)
        df = df.div(track["TOD"].diff(periods=1), axis = 0)
        df.dropna()
        return np.linalg.pinv(df.cov().values)
    except:
        print(df)
        print(df.cov())
        print(df.cov().values)
        print(np.linalg.pinv(df.cov().values))


def calculateCovarianceMatrix_H(track):
    df = track[["X", "Y", "Z", "TOD"]]
    df = df.loc[(df!=0).any(axis=1)]
    t = df[["X", "Y", "Z"]].diff(periods=1)
    t = t.div(df["TOD"].diff(periods=1), axis = 0)
    t.dropna()
    return np.linalg.pinv(t.cov().values)



def calculateAssociationScore(ini, middle, end, cov_inv, cov_inv_h):

    t_ratio = (middle["TOD"] - ini["TOD"]) / (end["TOD"] - ini["TOD"])
    estimated = pd.DataFrame()

    estimated["X"] = [ini["X"] + (end["X"]-ini["X"]) / t_ratio] #brackets necessary
    estimated["Y"] = ini["Y"] + (end["Y"]-ini["Y"]) / t_ratio

    if ini["Z"] == 0.0 or end["Z"] == 0.0 or middle["Z"] == 0.0:
        cov = cov_inv
        m = middle[["X", "Y"]].to_numpy()
    else:
        cov = cov_inv_h
        estimated["Z"] = ini["Z"] + (end["Z"]-ini["Z"]) / t_ratio
        m = middle[["X", "Y", "Z"]].to_numpy()

    #print(estimated)
    #print(ini[["X", "Y", "Z", "TOD"]])
    #print(end[["X", "Y", "Z", "TOD"]])
    try:
        e = estimated.to_numpy()[0]

        distance = np.matmul(np.transpose(m - e), cov)
        distance = np.matmul(distance, (m - e))
        distance = distance**(1/2)
    except ValueError:

        print("e: ", e)
        print("m: ", m)
        print("cov: ", cov)
        print("cov_inv: ", cov_inv)
        print("cov_inv_h: ", cov_inv_h)


        raise SystemExit("Stop right there!")

    return distance

def cleanCodeList(list):
    cleanList = [int(x) for x in list
                 if x is not None
                 and not np.isnan(x)
                 and x != 0]
    return cleanList

class CellList:
    def __init__(self):
        self.cellList = []

    def updateList(self, track):
        self.cellList = track[["X", "Y", "Z", "TOD"]].floordiv(50).value_counts().reset_index(name='counts')
        self.cellList.drop('counts', inplace=True, axis=1)

    def checkList(self, candCell):
        merged = pd.merge(self.cellList, candCell.cellList, on=["X", "Y", "Z", "TOD"], how='right', indicator='exists')
        merged['exists'] = np.where(merged.exists == 'both', True, False)

        return merged['exists'].any()


class Track:
    def __init__(self, track, tAddr = None):
        self.track = track
        self.track.sort_values(by="TOD")
        self.cellList = CellList()
        self.cellList.updateList(self.track)
        self.tAddr = tAddr
        self.cov_inv = []
        self.cov_inv_h = []

    def hasTAddr(self):
        if self.tAddr is None or \
                self.tAddr == "NULL" or \
                self.tAddr == 0 or \
                self.tAddr == "":
            return False
        return True

    def updateCovariances(self):
        if len(self.track) < 3:
            return
        self.cov_inv = calculateCovarianceMatrix(self.track)
        self.cov_inv_h = calculateCovarianceMatrix_H(self.track)
        if len(self.cov_inv) < 2:
            raise Exception("COV INV FAILURE")
        if len(self.cov_inv_h) < 3:
            raise Exception("COV H FAILURE")

    def checkCompatibilityBetweenTracks(self, candidate):

        if self.hasTAddr() and candidate.hasTAddr() and self.tAddr != candidate.tAddr:
            return np.inf
        tcan = candidate.track
        ttrack = self.track
        distance = []
        index = 0

        if len(self.track) < 3:
            dist = self.shortTrackCompatibilityCheck(candidate)
            #print(dist)
            return np.mean(dist)

        #Check time overlap
        if tcan["TOD"].max() < ttrack["TOD"].min() - 75 or tcan["TOD"].min() > ttrack["TOD"].max() + 75:
            return np.inf

        #if not self.cellList.checkList(candidate.cellList):
        #    return np.inf

        for _, tr in tcan.iterrows():

            while  (index < (ttrack.shape[0] - 2)) and ((ttrack.iloc[index+1])["TOD"] < tr["TOD"] + 0.5):
                index = index +1
            if (ttrack.iloc[index])["TOD"] > tr["TOD"] - 0.5:
                continue

            distance.append(calculateAssociationScore(ttrack.iloc[index], tr, ttrack.iloc[index+1], self.cov_inv, self.cov_inv_h))

        if len(distance) < 1:
            return np.inf
        return np.mean(distance)

    def shortTrackCompatibilityCheck(self, candidate):
        dt = abs(self.track["TOD"].values - candidate.track["TOD"].values)
        dist = ((self.track["X"].values - candidate.track["X"].values )**2 + (self.track["Y"].values - candidate.track["Y"].values)**2)**(1/2)

        pond_dist = dist * dt * 10

        return pond_dist


    def joinTracks(self, track):
        self.track = pd.concat([self.track, track.track],ignore_index=True)
        self.track.sort_values(by=["TOD"])
        self.updateCovariances()
        self.cellList.updateList(self.track)

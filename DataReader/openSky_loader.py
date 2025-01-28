
import pandas as pd
import sqlalchemy as db
import sys


def process_csv(filename):
    print("Reading CSV")
    data = pd.read_csv(filename)
    points = len(data)
    print(str(points) + " datapoints read")

    data.dropna(subset=["lat"], inplace=True)

    print(str(points - len(data)) + " datapoints dropped due to no pos")
    d = pd.DataFrame()

    d["TOD"] = data["time"]
    d["POS_LAT_DEG"] = data["lat"]
    d["POS_LONG_DEG"] = data["lon"]

    d["TARGET_ADDR"] = data["icao24"].apply(int, base=16)
    d["MODE3A_CODE"] = data["squawk"]

    d["onGround"] = data["onground"]
    d["vertRate"] = data["vertrate"]

    d["H_latlon"] = data["geoaltitude"]

    d.dropna(subset=["H_latlon"], inplace=True)

    d["DS_ID"] = str(9999)
    d["SIC"] = str(99)
    d["SAC"] = str(99)

    d["lastposupdate"] = data["lastposupdate"]
    
    return d


if __name__ == "__main__":

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        print("No filename provided")
        exit()

    data = process_csv(filename)



    print(data)
    print(data.shape)
    
    #connection = engine.connect()

    #data.to_sql("sensorD ata", con=engine, if_exists="replace")

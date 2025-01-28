from matplotlib import colors

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import time


# import gmplot package 
import gmplot 

m2feet = 3.28084
m2NM = 0.000539957
seconds_in_day = 86400


class Plotter:

    def plotTracks(self, track, candidate, distance = None, id = ""):
        f, (ax1, ax2) = plt.subplots(1, 2)

        x_1 = track["X"]
        y_1 = track["Y"]

        x_2 = candidate["X"]
        y_2 = candidate["Y"]

        ax1.plot(x_1, y_1, color='#00ff00', linestyle='--', marker='1')
        ax1.plot(x_2, y_2, color='#ff0000', linestyle='--', marker='2')

        ax1.legend(["Track","Candidate"], loc=1)
        ax1.set_title(str(id) + ' track')
        if distance is not None:
            ax2.plot(distance)
            ax2.set_ylim([0, 1500])
        plt.show(block=False)

    def plot_by_ID(self, track, id):
        track2 = track[track["TARGET_ADDR"] == id] 
        
        self.plotTracksTOD(track2)

    
        
    def plotTracksTOD(self, track, candidate = None, id = "", color=["#00ff00", "#0000ff"]):
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        x_1 = track["X"]
        y_1 = track["Y"]
        h_1 = track["Z"]
        tod_1 = track["TOD"]


        ax1.legend(["Track","Candidate"], loc=1)
        ax1.set_title(str(id) + ' track')

        ax1.plot(x_1, y_1, color=color[0], linestyle='None', marker='1')
        ax2.plot(tod_1, h_1, color=color[0], linestyle='None', marker='1')

        ax3.plot(tod_1, x_1, color=color[0], linestyle='None', marker='1')
        ax4.plot(tod_1, y_1, color=color[0], linestyle='None', marker='1')


        if candidate is not None:
            x_2 = candidate["X"]
            y_2 = candidate["Y"]
            h_2 = candidate["Z"]
            tod_2 = candidate["TOD"]

            ax1.plot(x_2, y_2, color=color[1], linestyle='None', marker='2')
            ax2.plot(tod_2, h_2, color=color[1], linestyle='None', marker='2')
            ax3.plot(tod_2, x_2, color=color[1], linestyle='None', marker='1')
            ax4.plot(tod_2, y_2, color=color[1], linestyle='None', marker='1')


        plt.show(block=False)

    def plotTracks_phase(self, track, candidate = None, id = "", color=["#00ff00", "#0000ff", "#ff0000"]):
        plt.figure(1)
        base = track[track["PHASE"] != "ML_PREDICTION"]
        ml = track[track["PHASE"] == "ML_PREDICTION"]
        
        x_1 = base["X"] * m2NM
        y_1 = base["Y"] * m2NM
        h_1 = base["Z"] * m2feet /2
        tod_1 = base["TOD"] % seconds_in_day

        plt.plot(x_1, y_1, color=color[0], linestyle='None', marker='1', label="Identified traffic")
        plt.plot(ml["X"]* m2NM, ml["Y"]* m2NM, color=color[1], linestyle='None', marker='1', label="Correct associations")


        plt.figure(2)
        
        plt.plot(tod_1, h_1, color=color[0], linestyle='None', marker='1', label="Identified traffic")
        plt.plot(ml["TOD"] % seconds_in_day  , ml["Z"]* m2feet/2, color=color[1], linestyle='None', marker='1', label="Correct associations")

        if candidate is not None:
            x_2 = candidate["X"]* m2NM
            y_2 = candidate["Y"]* m2NM
            h_2 = candidate["Z"]* m2feet/2
            tod_2 = candidate["TOD"] % seconds_in_day           
            
            plt.plot(tod_2, h_2, color=color[2], linestyle='None', marker='2', label="Incorrect associations")
            plt.figure(1)
            plt.plot(x_2, y_2, color=color[2], linestyle='None', marker='2', label="Incorrect associations")

        plt.xlabel("X (NM)")
        plt.ylabel("Y (NM)")
        plt.legend()
        path = 'img/' + str(id) + "H_" +  str(time.time()) 
        plt.savefig(path + ".eps", format='eps')
        plt.savefig(path + ".png", format='png')

        plt.figure(2)
        plt.xlabel("Time of Day (s)")
        plt.ylabel("Altitude (feet)")
        plt.legend()
        path = 'img/' + str(id) + "V_" +  str(time.time()) 
        plt.savefig(path + ".eps", format='eps')
        plt.savefig(path + ".png", format='png')
        plt.show()

        
    def plotTracks_phase_XY(self, track, candidate = None, id = "", color=["#00ff00", "#0000ff", "#ff0000"]):
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        base = track[track["PHASE"] != "ML_PREDICTION"]
        ml = track[track["PHASE"] == "ML_PREDICTION"]
        
        x_1 = base["X"]
        y_1 = base["Y"]
        h_1 = base["Z"]
        tod_1 = base["TOD"]


        ax1.legend(["Track","Candidate"], loc=1)
        ax1.set_title(str(id) + ' track')

        ax1.plot(x_1, y_1, color=color[0], linestyle='None', marker='1')
        ax1.plot(ml["X"], ml["Y"], color=color[1], linestyle='None', marker='1')

        ax2.plot(tod_1, h_1, color=color[0], linestyle='None', marker='1')
        ax2.plot(ml["TOD"], ml["Z"], color=color[1], linestyle='None', marker='1')

        ax3.plot(tod_1, x_1, color=color[0], linestyle='None', marker='1')
        ax4.plot(tod_1, y_1, color=color[0], linestyle='None', marker='1')
        ax3.plot(ml["TOD"], ml["X"], color=color[1], linestyle='None', marker='2')
        ax4.plot(ml["TOD"], ml["Y"], color=color[1], linestyle='None', marker='2')


        if candidate is not None:
            x_2 = candidate["X"]
            y_2 = candidate["Y"]
            h_2 = candidate["Z"]
            tod_2 = candidate["TOD"]

            ax1.plot(x_2, y_2, color=color[2], linestyle='None', marker='2')
            ax2.plot(tod_2, h_2, color=color[2], linestyle='None', marker='2')
            ax3.plot(tod_2, x_2, color=color[2], linestyle='None', marker='1')
            ax4.plot(tod_2, y_2, color=color[2], linestyle='None', marker='1')


        plt.show(block=False)

    def plotTracks_byDSID(self, track, candidate = None, id = "", color=["#00ff00", "#0000ff"]):
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        x_1 = track["X"]
        y_1 = track["Y"]
        h_1 = track["Z"]
        tod_1 = track["TOD"]
        DSID_1 = track["DS_ID"]


        # Get unique sources
        unique_sources = track['DS_ID'].unique()
        
        # Create a color map for sources
        color_map = plt.cm.get_cmap('YlGn', len(unique_sources))
        
        # Create a scatterplot for each source with different colors
        for i, source in enumerate(unique_sources):
            source_data = track[track['DS_ID'] == source]
            ax1.scatter(source_data['X'], source_data['Y'], label=f'Source {source}', c=[color_map(i)])#, marker='1')
        
        # Add legend
        plt.legend()

        #ax1.legend(["Track","Candidate"], loc=1)
        #ax1.set_title(str(id) + ' track')

        #ax1.plot(x_1, y_1, color=color[0], linestyle='None', marker='1')
        ax2.plot(tod_1, h_1, color=color[0], linestyle='None', marker='1')

        ax3.plot(tod_1, x_1, color=color[0], linestyle='None', marker='1')
        ax4.plot(tod_1, y_1, color=color[0], linestyle='None', marker='1')


        if candidate is not None:
            x_2 = candidate["X"]
            y_2 = candidate["Y"]
            h_2 = candidate["Z"]
            tod_2 = candidate["TOD"]
            DSID_2 = track["DS_ID"]

            #ax1.plot(x_2, y_2, color=DSID_2, linestyle='None', marker='2')
                    # Get unique sources
            unique_sources = track['DS_ID'].unique()
            
            # Create a color map for sources
            color_map = plt.cm.get_cmap('PuRd', len(unique_sources))
            
            # Create a scatterplot for each source with different colors
            for i, source in enumerate(unique_sources):
                source_data = candidate[candidate['DS_ID'] == source]
                ax1.scatter(source_data['X'], source_data['Y'], label=f'Source {source}', c=[color_map(i)])#, marker='2')
        
            ax2.plot(tod_2, h_2, color=color[1], linestyle='None', marker='2')
            ax3.plot(tod_2, x_2, color=color[1], linestyle='None', marker='2')
            ax4.plot(tod_2, y_2, color=color[1], linestyle='None', marker='2')


        plt.show(block=False)


    def plotTracksH(self, track, candidate = None, id = "", color=["#00ff00", "#0000ff"]):
        f, (ax1, ax2) = plt.subplots(1, 2)

        x_1 = track["X"]
        y_1 = track["Y"]
        h_1 = track["Z"]
        tod_1 = track["TOD"]


        ax1.legend(["Track","Candidate"], loc=1)
        ax1.set_title(str(id) + ' track')

        ax1.plot(x_1, y_1, color=color[0], linestyle='None', marker='1')
        ax2.plot(tod_1, h_1, color=color[0], linestyle='None', marker='1')

        if candidate is not None:
            x_2 = candidate["X"]
            y_2 = candidate["Y"]
            h_2 = candidate["Z"]
            tod_2 = candidate["TOD"]

            ax1.plot(x_2, y_2, color=color[1], linestyle='None', marker='2')
            ax2.plot(tod_2, h_2, color=color[1], linestyle='None', marker='2')


        plt.show(block=False)

    def plotGmaps(self, row, id=""):
        gmap5 = gmplot.GoogleMapPlotter(41.2986291691189, 2.08024592338138, 10, apikey = apikey) 
          
        gmap5.scatter( row["POS_LAT_DEG"], row["POS_LONG_DEG"], '# FF0000', 
                                        size = 40, marker = False) 
                
        gmap5.draw( "plots/trajectory_" + str(id) + ".html" ) 
    
    def plotGmaps_id(self, correct, incorrect, id=""):
        gmap5 = gmplot.GoogleMapPlotter(41.2986291691189, 2.08024592338138, 10, apikey = apikey) 
          
        gmap5.scatter( correct["POS_LAT_DEG"], correct["POS_LONG_DEG"], "#F0FF00", 
                                        size = 40, marker = False) 
        gmap5.scatter( incorrect["POS_LAT_DEG"], incorrect["POS_LONG_DEG"], "#FF0000", 
                                        size = 40, marker = False) 
                
        gmap5.draw( "plots/trajectory_color" + str(id) + ".html" ) 
  
  
    def plotGmapsPlotLines(self, lines, id=""):
        gmap5 = gmplot.GoogleMapPlotter(41.2986291691189, 2.08024592338138, 10, apikey = apikey)

        for row in lines:
            gmap5.plot( row["POS_LAT_DEG"], row["POS_LONG_DEG"], '#FF0000',
                           size = 40, marker = False)

        gmap5.draw( "plots/trajectory_lines" + id + ".html" )


    def plotLatLon(self, track, candidate):
        x_1 = track["POS_LAT_DEG"]
        y_1 = track["POS_LONG_DEG"]

        x_2 = candidate["POS_LAT_DEG"]
        y_2 = candidate["POS_LONG_DEG"]

        plt.plot(x_1, y_1, color='#00ff00', linestyle='--', marker=',')
        plt.plot(x_2, y_2, color='#ff0000', linestyle='--', marker=',')
        plt.legend(["Track latlon","Candidate latlon"], loc=1)

        plt.show(block= False)


    def plotAllTracks(self, tracklist, id = "", color=["#00ff00", "#0000ff"]):
        f, (ax1, ax2) = plt.subplots(1, 2)
        color_id = 0
        for t in tracklist:
            track = t.track
            x_1 = track["X"]
            y_1 = track["Y"]
            h_1 = track["H_latlon"]
            tod_1 = track["TOD"]
            color = list(colors.TABLEAU_COLORS.values())[color_id]
            color_id = (color_id + 1) % len(colors.TABLEAU_COLORS)

            ax1.plot(x_1, y_1, color=color, linestyle='None', marker='1')
            ax2.plot(tod_1, h_1, color=color, linestyle='None', marker='2')

        plt.show(block=False)

    def plotAllTracks_fading(self, tracklist, TOD, threshold=30, id = "",x_lim=None, y_lim=None):
        f, (ax1, ax2) = plt.subplots(1, 2)
        color_id = 0
        for t in tracklist:
            track = t.points
            #filter out tracks that are in the 30sec range
            track = track[np.abs(track["TOD"]-TOD)<threshold+threshold/2]
            if len(track) < 4:
                continue
            x_1 = track["X"]
            y_1 = track["Y"]
            h_1 = track["Z"]
            tod_1 = track["TOD"]
            alpha = 1 - np.abs(tod_1-TOD)/threshold
            alpha = alpha.clip(0.05,1)
            
            alpha = alpha.to_numpy()

            color = list(colors.TABLEAU_COLORS.values())[color_id]
            color_id = (color_id + 1) % len(colors.TABLEAU_COLORS)
            #print(t.taddr)

            bad = track[track["original_Taddr"].astype(float) != float(t.taddr)]
            ax1.scatter(x_1, y_1, color=color, alpha=alpha, linestyle='None', marker='.')
            ax2.scatter(tod_1, h_1, color=color,alpha=alpha, linestyle='None', marker='.')
            ax1.scatter(bad["X"], bad["Y"], color="#ff0000",  linestyle='None', marker='.')

        if x_lim is not None:
            ax1.set_xlim(x_lim)
        if y_lim is not None:
            ax1.set_ylim(y_lim)

        
        plt.show(block=False)
    
    def plotAllTracks_fading_eps(self, tracklist, TOD, threshold=30, id = "",x_lim=None, y_lim=None):
        ax = plt.gca()
        color_id = 0
        for t in tracklist:
            track = t.points
            #filter out tracks that are in the 30sec range
            track = track[np.abs(track["TOD"]-TOD)<threshold+threshold/2]
            if len(track) < 4:
                continue

            
            
            x_1 = track["X"] * m2NM
            y_1 = track["Y"] * m2NM
            h_1 = track["Z"] * m2feet
            tod_1 = track["TOD"] % seconds_in_day

            alpha = 1 - np.abs(tod_1-TOD)/threshold
            alpha = alpha.clip(0.05,1)
            
            alpha = alpha.to_numpy()

            color = list(colors.TABLEAU_COLORS.values())[color_id]
            color_id = (color_id + 1) % len(colors.TABLEAU_COLORS)
            #print(t.taddr)

            bad = track[track["original_Taddr"].astype(float) != float(t.taddr)]
            plt.scatter(x_1, y_1, color=color, alpha=alpha, linestyle='None', marker='.')
            plt.scatter(bad["X"] * m2NM, bad["Y"] * m2NM, color="#ff0000", alpha=alpha, linestyle='None', marker='.')

        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)

        plt.xlabel("X (NM)")
        plt.ylabel("Y (NM)")
        path = 'img/' + str(id) + "H_" +  str(time.time()) 
        plt.savefig(path + ".eps", format='eps')
        plt.savefig(path + ".png", format='png')

        plt.show()

def plotBins(list_datasets):
    for dataset in list_datasets:
        sns.histplot(data=dataset, discrete=True, stat='count', bins=2)
        
    plt.show()

def plotCategories(list_datasets):
    for dataset in list_datasets:
        dataset['Category'] = 'Compatible' if i % 2 == 0 else 'Non-Compatible'
    sns.histplot(data=dataset, hue='Category', discrete=True, stat='count', multiple='stack', bins=2)
        
    plt.show()

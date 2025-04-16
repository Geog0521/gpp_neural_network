##Creator: Xinghua Cheng
##Contact: xinghua.cheng@uconn.edu
##Date: April 15, 2024

##Goal: to prepare data for the neural network input
import os
import pandas as pd

def prepare_data_neural_network(dataFolder,dadata_train_tab):
    ## read .tif files and get the year with the naming rule "HLS.L30.T06WVS.2019186T211232.v2.0.US-BZo_merged.50x50pixels.tiff"
    trainFolder = dataFolder + "\\" + "train"
    tiffLis = os.listdir(trainFolder)
    df = pd.read_csv(dataFolder + "\\"+dadata_train_tab)

    # Initialize list to store matched results
    all_results = []

    for chipTif in tiffLis:
        print("We are reading " + chipTif)
        yearStr = chipTif.split(".")[3]
        year = int(yearStr[0:4])
        print("The year is :{}".format(year))

        ## Filter based on a specific column value
        ## Note: the suffix of downloaded file is .tiff, while that of file name in "Chip" column is ".tif"
        ## We get T2MIN, T2MAX, T2MEAN, TSMDEWMEAN, GWETROOT, LHLAND, SHLAND, SWLAND, PARDFLAND, PRECTOTLAND, GPP, date, year, month, doy
        result = df[df["Chip"] == chipTif[:-1]][["T2MIN", "T2MAX", "T2MEAN", "TSMDEWMEAN", "GWETROOT", "LHLAND", "SHLAND", "SWLAND", "PARDFLAND", "PRECTOTLAND", "GPP", "date", "year", "month", "doy"]]
        print(result)

        if not result.empty:
            result["Chip"] = chipTif
            all_results.append(result)
        else:
            print(f"Warning: No match found for {chipTif}")

    # Concatenate all filtered rows into one DataFrame
    combined_df = pd.concat(all_results, ignore_index=True)
    savedCSV = dataFolder + "\\"+"hls_gpp_envi.csv"
    combined_df.to_csv(savedCSV, index=False)
    print("Please go to :{} and move to Step_2".format(savedCSV))
    del df

    return 1

if __name__ == "__main__":
    print("We are preparing data")
    dataFolder = "E:\\GPP_exercise\\hls_merra2_gppFlux"
    dadata_train_tab = "data_train_hls_37sites_v0_1.csv"
    prepare_data_neural_network(dataFolder,dadata_train_tab)
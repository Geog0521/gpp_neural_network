##Creator: Xinghua Cheng
##Contact: xinghua.cheng@uconn.edu
##Date: April 16, 2025

##Goal: to prepare test dataset

import numpy as np
import pandas as pd
from tqdm  import tqdm
import os
from osgeo import gdal


def prepare_data_test(dataFolder,dadata_train_tab,savedCSV="hls_gpp_envi_test.csv"):


    ## read .tif files and get the year with the naming rule "HLS.L30.T06WVS.2019186T211232.v2.0.US-BZo_merged.50x50pixels.tiff"
    trainFolder = dataFolder + "\\" + "test"
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
    savedCSV = dataFolder + "\\"+savedCSV
    combined_df.to_csv(savedCSV, index=False)

    del df

    return 1
##
def convert_data_to_npz(dataFolder,hls_gpp_evs_cvs,folder="test"):
    # Containers
    X_img_list = []
    X_env_list = []
    y_list = []

    df = pd.read_csv(dataFolder+"\\"+hls_gpp_evs_cvs)
    for idx,row in tqdm(df.iterrows(),total=len(df)):
        hls_path = os.path.join(dataFolder+"\\"+folder+"\\"+row["Chip"])
        ds = gdal.Open(hls_path)
        hlsimg = np.stack([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(ds.RasterCount)])
        X_img_list.append(hlsimg.astype(np.float32))
        #obtain the environment variables
        evs = row[["T2MIN", "T2MAX", "T2MEAN", "TSMDEWMEAN", "GWETROOT", "LHLAND", "SHLAND", "SWLAND", "PARDFLAND", "PRECTOTLAND"]].values
        evs = evs.astype(np.float32)
        X_env_list.append(evs)

        ## the target variable
        gpp = df.loc[idx, 'GPP'].astype(np.float32)
        y_list.append(gpp)
        del ds

    # Convert to arrays
    X_img = np.stack(X_img_list)  # shape: (N, 6, 50, 50)
    X_env = np.stack(X_env_list)  # shape: (N, 10)
    y = np.array(y_list)  # shape: (N,)

    print("HLS image shape:", X_img.shape)  # (N, 6, 50, 50)
    print("MERRA aux shape:", X_env.shape)  # (N, 10)
    print("GPP target shape:", y.shape)  # (N,)

    # Assuming you also have a 'year' list aligned with your samples
    year_list = df['year'].values  # or merra_df['year'].values
    print("year list:", year_list)
    np.savez_compressed("data/gpp_dataset_"+folder+".npz",
                        X_img=X_img,
                        X_aux=X_env,
                        y=y,
                        year=year_list)

    return 1

if __name__ == "__main__":
    dataFolder = "E:\\GPP_exercise\\hls_merra2_gppFlux"
    dadata_train_tab = "data_train_hls_37sites_v0_1.csv"
    hls_gpp_evs_file = "hls_gpp_envi_test.csv"
    prepare_data_test(dataFolder,dadata_train_tab,savedCSV=hls_gpp_evs_file)
    print("We are converting data to .npz")
    convert_data_to_npz(dataFolder, hls_gpp_evs_file,folder="test")

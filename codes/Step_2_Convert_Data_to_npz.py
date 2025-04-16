import numpy as np
import pandas as pd
from tqdm  import tqdm
import os
from osgeo import gdal

##
def convert_data_to_npz(dataFolder,hls_gpp_evs_cvs):
    # Containers
    X_img_list = []
    X_env_list = []
    y_list = []

    df = pd.read_csv(dataFolder+"\\"+hls_gpp_evs_cvs)
    for idx,row in tqdm(df.iterrows(),total=len(df)):
        hls_path = os.path.join(dataFolder+"\\"+"train"+"\\"+row["Chip"])
        ds = gdal.Open(hls_path)
        hlsimg = np.stack([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(ds.RasterCount)])
        X_img_list.append(hlsimg.astype(np.float32))

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
    np.savez_compressed("data/gpp_dataset.npz",
                        X_img=X_img,
                        X_aux=X_env,
                        y=y,
                        year=year_list)

    return 1

if __name__ == "__main__":
    dataFolder = "E:\\GPP_exercise\\hls_merra2_gppFlux"
    hls_gpp_evs_file = "hls_gpp_envi.csv"
    print("We are converting data to .npz")
    yearLis = [2018,2019,2021]
    convert_data_to_npz(dataFolder, hls_gpp_evs_file)
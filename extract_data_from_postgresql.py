import os
import folium
import pandas as pd
import json
import psycopg2
import pickle
import numpy as np
from haversine import haversine
import sqlalchemy
from sqlalchemy.sql import text
from tqdm import tqdm
from datetime import datetime
from time import time
from zipfile import ZipFile
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

def cov(df):

    df = df.astype({'latitude':float, 'longitude':float})
    x = df.loc[:, ['latitude', 'longitude']]
    x = np.array(x)
    m = [x[0], x[int((len(x)-1)/2)], x[len(x)-1]]
    m = np.array(m)
    m_mu = np.mean(m, axis=0)
    sb = (m-m_mu).T @ (m-m_mu)

    mu = np.mean(x, axis=0)
    covar = (x-mu).T@(x-mu)
    W = covar@sb

    w, v = np.linalg.eig(W)
    v0 = [v[0, 0], v[1, 0]]
    v1 = [v[0, 1], v[1, 1]]

    if w[0] > w[1]:
        pc = np.array(v0)
    else:
        pc = np.array(v1)
    x = pc @ x.T
    mini = np.min(x)
    maxi = np.max(x)
    nor_X = (x - mini) / (maxi - mini)

    return nor_X

def making_unique_port(tot_df, new_df):

    columns = list(new_df.keys())
    port_df = pd.DataFrame(columns=columns)
    port_lsts = sorted(set(new_df['dt_pos_utc']))
    dicts = {}
    jdx = 0
    for port_lst in port_lsts:
        dicts[port_lst] = []
        for i in range(new_df.shape[0]):
            if port_lst == new_df['dt_pos_utc'][i]:
                dicts[port_lst].append(new_df.loc[i])

        if len(dicts[port_lst]) == 1:
            port_df.loc[jdx] = dicts[port_lst][0]
            jdx += 1

        elif len(dicts[port_lst]) > 1:
            mini = 100000
            data = dicts[port_lst]


            for j in range(len(data)):
                distance = haversine((data[j][1], data[j][2]), (data[j][3], data[j][4]), unit='km')
                if mini > distance:
                    mini = distance
                    k = j
            port_df.loc[jdx] = data[k]
            jdx += 1
    port_df = port_df.sort_values('dt_pos_utc')


    atd_lsts = []
    ata_lsts = []
    atb_lsts = []
    for i in range(port_df.shape[0]):

        jdx = tot_df.index[tot_df['dt_pos_utc'] == port_df['dt_pos_utc'][i]]
        k = list(jdx)[0] + 5
        l = list(jdx)[0] + 5

        while 1:
            k += 1

            if tot_df['heading'][k] <= port_df['heading'][i]-2 \
                    or tot_df['heading'][k] >= port_df['heading'][i]+2:
                # print("atd", k)
                # print(tot_df['heading'][k])
                atd_lsts.append([tot_df.loc[k-1, 'latitude'], tot_df.loc[k-1, 'longitude'],
                                 tot_df.loc[k-1, 'dt_pos_utc']])
                break
            elif k == tot_df.shape[0] - 1:
                atd_lsts.append([tot_df.loc[k, 'latitude'], tot_df.loc[k, 'longitude'],
                                 tot_df.loc[k, 'dt_pos_utc']])
                break


        while 1:
            l -= 1
            # print("ata", l)
            if tot_df['heading'][l] <= port_df['heading'][i]-2 \
                    or tot_df['heading'][l] >= port_df['heading'][i]+2:
                ata_lsts.append([tot_df.loc[l+1, 'latitude'], tot_df.loc[l+1, 'longitude'],
                                 tot_df.loc[l+1, 'dt_pos_utc']])
                break
            elif l == 0:
                ata_lsts.append([tot_df.loc[l, 'latitude'], tot_df.loc[l, 'longitude'],
                                 tot_df.loc[l, 'dt_pos_utc']])
                break


        m = l
        if m == 0:
            atbs = [np.array(tot_df.loc[m, 'latitude']),
                    np.array(tot_df.loc[m, 'longitude']),
                    np.array(tot_df.loc[m, 'dt_pos_utc'])]
        else:
            while 1:
                m -= 1
                if int(tot_df['sog'][m]) >= 10:
                    break
                if m == 0:
                    break
            tmp_df = tot_df.loc[m:l+50, ['latitude', 'longitude', 'sog', 'dt_pos_utc']]
            tmp_df.reset_index(drop=True, inplace=True)
            tmp_df = tmp_df.sort_values('dt_pos_utc')
            x = cov(tmp_df)

            if 1.0 in x:
                find_idx = np.where(x==1.0)
                atbs = [np.array(tmp_df.loc[find_idx, 'latitude'])[0],
                        np.array(tmp_df.loc[find_idx, 'longitude'])[0],
                        np.array(tmp_df.loc[find_idx, 'dt_pos_utc'])[0]]

            else:
                atbs = [np.array(tmp_df.loc[0, 'latitude'])[0],
                        np.array(tmp_df.loc[0, 'longitude'])[0],
                        np.array(tmp_df.loc[0, 'dt_pos_utc'])[0]]

        atb_lsts.append(atbs)

    port_df = port_df.rename(columns={'dt_pos_utc': 'ATA'})

    ata_lat = []
    ata_long = []
    atas = []
    atb_lat = []
    atb_long = []
    atbs= []
    atd_lat = []
    atd_long = []
    atds = []

    for ata, atb, atd in zip(ata_lsts, atb_lsts, atd_lsts):

        ata_lat.append(ata[0])
        ata_long.append(ata[1])
        atas.append(ata[2])
        atb_lat.append(atb[0])
        atb_long.append(atb[1])
        atbs.append(atb[2])
        atd_lat.append(atd[0])
        atd_long.append(atd[1])
        atds.append(atd[2])
    # print(atds)
    # exit()
    port_df['ATD_lat'] = atd_lat
    port_df['ATD_long'] = atd_long
    port_df['ATD'] = atds
    port_df['ATA_lat'] = ata_lat
    port_df['ATA_long'] = ata_long
    port_df['ATA'] = atas
    port_df['ATB_lat'] = atb_lat
    port_df['ATB_long'] = atb_long
    port_df['ATB'] = atbs

    port_df = port_df[['port_code', 'latitude', 'longitude',
                       'ATA_lat', 'ATA_long', 'ATA',
                       'ATB_lat', 'ATB_long','ATB',
                       'ATD_lat', 'ATD_long', 'ATD']]
    port_df['ATB'] = np.where(port_df['ATA'] > port_df['ATB'], port_df['ATA'], port_df['ATB'])
    port_df['ATB_lat'] = np.where(port_df['ATA'] > port_df['ATB'], port_df['ATA_lat'], port_df['ATB_lat'])
    port_df['ATB_long'] = np.where(port_df['ATA'] > port_df['ATB'], port_df['ATA_long'], port_df['ATB_long'])

    port_df.to_csv(f"port_data/new_unique_{mmsi}_ports.csv", index=False, encoding='latin-1')

def extract_special_mmsi_data_from_postgresql(mmsi):

    url = 'postgresql+psycopg2://postgres:math3039@localhost:5432/postgres'
    engine = sqlalchemy.create_engine(url)

    # sql = f"""select * from hmm_sample_ship_data where mmsi = '{mmsi}' order by pot_utc;"""
    sql1 = f"""select distinct B.port_code, B.latitude as lat, B.longitude as lon, A.latitude, A.longitude, A.heading, A.sog, A.dt_pos_utc
                from (select sog, latitude, longitude, heading, nav_status, dt_pos_utc from public.hmm_10_ship_imo_mmsi_data where sog < 0.1 and mmsi='{mmsi}') as A
                        left outer join (select port_code, latitude, longitude from public.final_port_code_dictionary) as B
					    on (A.latitude - B.latitude)*(A.latitude - B.latitude)+(A.longitude - B.longitude)*(A.longitude - B.longitude) < ((1.0/380.0)*50)*((1.0/380.0)*50)
                where B.port_code is not NULL;"""

    sql2 = f"""select * from public.hmm_10_ship_imo_mmsi_data where mmsi='{mmsi}'"""

    with engine.connect().execution_options(autocommit=True) as conn:
        query1 = conn.execute(text(sql1))
        query2 = conn.execute(text(sql2))

    tot_df = pd.DataFrame(query2.fetchall())
    tot_df.sort_values('dt_pos_utc')

    df = pd.DataFrame(query1.fetchall())
    df = df.sort_values('dt_pos_utc')

    new_df = pd.DataFrame(columns=df.keys())
    new_df.loc[0] = df.loc[0]
    idx = 1
    for i in range(df.shape[0]-1):
        if df['port_code'][i] == df['port_code'][i+1]:
            continue
        elif df['port_code'][i] != df['port_code'][i+1]:
            new_df.loc[idx] = df.loc[i+1]
            idx+=1


    new_df = new_df.sort_values('dt_pos_utc')

    new_df.reset_index(drop=True, inplace=True)


    making_unique_port(tot_df, new_df)


def mmsi_location_map(coordinates, path, mmsi):

    url = 'postgresql+psycopg2://postgres:math3039@localhost:5432/postgres'
    engine = sqlalchemy.create_engine(url)

    sql = f"""select * from public.hmm_10_ship_imo_mmsi_data where mmsi='{mmsi}'"""

    with engine.connect().execution_options(autocommit=True) as conn:
        query = conn.execute(text(sql))

    df = pd.DataFrame(query.fetchall())

    df = df.sort_values('dt_pos_utc')
    df.reset_index(drop=True, inplace=True)

    seaMap = folium.Map(location=[35.95, 127.7], zoom_start=2)

    # for i in range(df.shape[0]):
    #     if 1:
    #         if df['longitude'][i] < -50:
    #             marker = folium.Circle(location=[df['latitude'][i], df['longitude'][i] + 360],
    #                                    fill='blue')
    #         else:
    #             marker = folium.Circle(location=[df['latitude'][i], df['longitude'][i]],
    #                                    fill='blue')
    #         marker.add_to(seaMap)

    for coord in coordinates:
        if coord[1] < -50:
            marker = folium.Circle(location=[coord[0], coord[1]+360],
                                   fill='red')

            marker_ata = folium.Marker(location=[coord[3], coord[4] + 360],
                                   tooltip='ATA',
                                   icon=folium.Icon(color='orange', icon='star')
                                   )
            marker_atb = folium.Marker(location=[coord[5], coord[6] + 360],
                                       tooltip='ATB',
                                       icon=folium.Icon(color='orange', icon='star')
                                       )
            marker_atd = folium.Marker(location=[coord[7], coord[8] + 360],
                                       tooltip='ATD',
                                       icon=folium.Icon(color='orange', icon='star')
                                       )
        else:
            marker = folium.Circle(location=[coord[0], coord[1]],
                                   fill='red')

            marker_ata = folium.Marker(location=[coord[3], coord[4]],
                                       tooltip='ATA',
                                       icon=folium.Icon(color='orange', icon='star')
                                       )
            marker_atb = folium.Marker(location=[coord[5], coord[6]],
                                       tooltip='ATB',
                                       icon=folium.Icon(color='orange', icon='star')
                                       )
            marker_atd = folium.Marker(location=[coord[7], coord[8]],
                                       tooltip='ATD',
                                       icon=folium.Icon(color='orange', icon='star')
                                       )
        marker.add_to(seaMap)
        marker_ata.add_to(seaMap)
        marker_atb.add_to(seaMap)
        marker_atd.add_to(seaMap)

    seaMap.save(os.path.join(path, f"loadMap_{mmsi}.html"))


def main(mmsi):

    start = datetime.fromtimestamp(time())
    extract_special_mmsi_data_from_postgresql(mmsi)
    end = datetime.fromtimestamp(time())
    print("elapsed time: {}".format(str(end - start)))

    path2 = 'port_data/'
    filenames2 = os.listdir(path2)
    if '.DS_Store' in filenames2:
        filenames2.remove('.DS_Store')
    coordinates = []
    for file in filenames2:
        if f'new_unique_{mmsi}_ports' in file:
            df = pd.read_csv(os.path.join(path2, file))
            for i in range(df.shape[0]):
                coordinates.append([float(df.loc[i, 'latitude']),
                                    float(df.loc[i, 'longitude']),
                                    df.loc[i,'port_code'],
                                    df.loc[i, 'ATA_lat'],
                                    df.loc[i, 'ATA_long'],
                                    df.loc[i, 'ATB_lat'],
                                    df.loc[i, 'ATB_long'],
                                    df.loc[i, 'ATD_lat'],
                                    df.loc[i, 'ATD_long']])

    mmsi_location_map(coordinates, 'map/', mmsi)

if __name__ == "__main__":

    #
    mmsis = [441111000, 441981000, 636020383, 636020615, 538007479, 370181000,
             416468000, 636092775, 564442000, 232031318]
    mmsis = sorted(mmsis)
    # mmsis = mmsis[7:]
    # exit()

    for mmsi in mmsis:
        main(mmsi)





    # cpu = os.cpu_count()
    # pool = Pool(cpu)
    # pool.map(make_dataFrame_to_csv, fileLists)
    # pool.close()
    # pool.join()







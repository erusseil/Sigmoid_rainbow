import io
import os
import requests
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import isfile, join


def extract_history(history_list: list, field: str) -> list:
    """Extract the historical measurements contained in the alerts
    for the parameter `field`.

    Parameters
    ----------
    history_list: list of dict
        List of dictionary from alert[history].
    field: str
        The field name for which you want to extract the data. It must be
        a key of elements of history_list
    
    Returns
    ----------
    measurement: list
        List of all the `field` measurements contained in the alerts.
    """
    if history_list is None:
        return []
    try:
        measurement = [obs[field] for obs in history_list]
    except KeyError:
        print('{} not in history data'.format(field))
        measurement = []

    return measurement

def extract_field(alert: dict, category: str, field: str) -> np.array:
    """ Concatenate current and historical observation data for a given field.
    
    Parameters
    ----------
    alert: dict
        Dictionnary containing alert data
    category: str
        prvDiaSources or prvDiaForcedSources
    field: str
        Name of the field to extract.
    
    Returns
    ----------
    data: np.array
        List containing previous measurements and current measurement at the
        end. If `field` is not in the category, data will be
        [alert['diaSource'][field]].
    """
    data = np.concatenate(
        [
            [alert["diaSource"][field]],
            extract_history(alert[category], field)
        ]
    )
    return data



def aggregate_columns(ftransfer_path, target):

    path = f'{ftransfer_path}/classId={target}/'
    all_files = [f for f in listdir(path)]
    all_formated = []

    for file in all_files:

        pdf = pd.read_parquet(f'{path}{file}')

        pdf['cflux'] = pdf[['diaSource', 'prvDiaForcedSources']]\
            .apply(lambda x: extract_field(x, 'prvDiaForcedSources', 'psFlux'), axis=1)
        pdf['csigflux'] = pdf[['diaSource', 'prvDiaForcedSources']]\
            .apply(lambda x: extract_field(x, 'prvDiaForcedSources', 'psFluxErr'), axis=1)
        pdf['cfid'] = pdf[['diaSource', 'prvDiaForcedSources']]\
            .apply(lambda x: extract_field(x, 'prvDiaForcedSources', 'filterName'), axis=1)
        pdf['cjd'] = pdf[['diaSource', 'prvDiaForcedSources']]\
            .apply(lambda x: extract_field(x, 'prvDiaForcedSources', 'midPointTai'), axis=1)

        pdf['ra'] = pdf['diaObject'].apply(lambda x: x['ra'])
        pdf['dec'] = pdf['diaObject'].apply(lambda x: x['decl'])
        pdf['hostgal_ra'] = pdf['diaObject'].apply(lambda x: x['hostgal_ra'])
        pdf['hostgal_dec'] = pdf['diaObject'].apply(lambda x: x['hostgal_dec'])
        pdf['hostgal_zphot'] = pdf['diaObject'].apply(lambda x: x['hostgal_zphot'])
        pdf['hostgal_zphot_err'] = pdf['diaObject'].apply(lambda x: x['hostgal_zphot_err'])
        pdf['objectId'] = pdf['diaObject'].apply(lambda x: x['diaObjectId'])

        cols = ['objectId', 'alertId', 'cjd', 'cflux', 'csigflux', 'cfid', 'ra', 'dec', 'hostgal_ra', 'hostgal_dec', 'hostgal_zphot', 'hostgal_zphot_err']
        formated = pdf[cols]
        all_formated.append(formated)

    final = pd.concat(all_formated, ignore_index=True)

    if not os.path.exists(f'{ftransfer_path}/aggregated/'):
        os.makedirs(f'{ftransfer_path}/aggregated/')

    final.to_parquet(f'{ftransfer_path}/aggregated/classId={target}.parquet')


    
    
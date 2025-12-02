import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.ned_calculator import NedCalculator


def prepare_data(columns, restricted=False, pc=False, mix_datasets=True, test_size=0.2, random_state=1):
    
    if restricted:
        test_df = pd.read_csv('refitting_results/output.csv') #### version with restricted reanalysed INSPIRE catalogue - will need a lot of work, defo not ready for use i think. 
        train_df = pd.read_csv('data/E-INSPIRE_I_master_catalogue.csv')
        train_df['logAge'] = np.log10(train_df['age_mean_mass']) + 9 # 9 is the Gyr conversion I think 
        train_mapping = {
                'logAge': 'logAge',
                'age_err_mass': 'age_err',
                'meanRadkpc_r': 'rad_kpc',
                'logM*': 'logM',
                'velDisp_ppxf_res': 'vdisp',
                '[M/H]_mean_mass': 'met',
                '[M/H]_err_mass': 'met_err'} # what about mgfe??
    else:
        test_df = pd.read_csv('data/INSPIRE_stelpop.csv')  # contains full test data
        
        train_df = pd.read_csv('data/stel_pop_fit_allZ.csv')  # has m/h and age and errors
        restricted_df =  pd.read_csv('data/E-INSPIRE_I_master_catalogue.csv')
        restricted_cols = ['GALAXY ID','univ_age', 'MgFe', 'velDisp_ppxf_res', 'DoR', 'logM*', 'meanRadkpc_kids', 'age_err_mass', 'SNR', '[M/H]_mean_mass']
        restricted_df = restricted_df[restricted_cols]
        
        dor_df = pd.read_csv('data/ppxf_stel_pop_test2_small.csv')
        dor_df.rename(columns={'dor_2':'dor_4'}, inplace=True)
        restricted_df.rename(columns={'DoR':'dor_26'}, inplace=True)
        
        restricted_df.sort_values(by='[M/H]_mean_mass', ascending=False, inplace=True)
        dor_df.sort_values(by='[M/H]_mean_1', ascending=False, inplace=True)
        
        restricted_df['[M/H]_mean_mass_rounded'] = restricted_df['[M/H]_mean_mass'].round(5)
        dor_df['[M/H]_mean_1_rounded'] = dor_df['[M/H]_mean_1'].round(5)
        
        restricted_df = restricted_df.merge(
            dor_df, 
            left_on='[M/H]_mean_mass_rounded',
            right_on='[M/H]_mean_1_rounded',
            how='left'
        )
        
        restricted_df = restricted_df.drop(['[M/H]_mean_mass_rounded', '[M/H]_mean_1_rounded'], axis=1)
        
        train_df = train_df.merge(
            restricted_df, 
            left_on='sexa_id',
            right_on='GALAXY ID', 
            how='inner'
        )
        
        train_df.rename(columns={'sexa_id':'ID'}, inplace=True)
        
        """train_df = pd.read_csv('../data/stel_pop_fit_allZ.csv')  # has m/h and age and errors
        restricted_df =  pd.read_csv('../data/E-INSPIRE_I_master_catalogue.csv') # contains below columns we need to extract
        restricted_cols = ['GALAXY ID','univ_age', 'MgFe', 'velDisp_ppxf_res', 'DoR', 'logM*', 'meanRadkpc_kids', 'age_err_mass', 'SNR'] # from unrestricted catalogue
        
        restricted_df=restricted_df[restricted_cols]
        train_df = train_df.merge(restricted_df, left_on='sexa_id',right_on='GALAXY ID', how='inner')"""
        
        age_columns = ['age_noboot', 'age_boot', 'age_minus', 'ages_plus']
        
        # train_df['logAge_err'] = train_df[age_columns].std(axis=1)
        train_df['logAge_err'] = np.log10(train_df['age_err_mass'])+9
        train_df['logAge'] = train_df[age_columns].mean(axis=1)
        
        train_df['lin_age'] = 10**(train_df['logAge']-9)
        train_df['lin_age_err'] = train_df['age_err_mass']
    
        
        metal_columns = ['metals_noboot', 'metals_boot', 'metals_minus', 'metal_plus']
        train_df['met_err'] = train_df[metal_columns].std(axis=1)
        train_df['met'] = train_df[metal_columns].mean(axis=1)
            
        train_mapping = {
                #'logAge': 'logAge',
                'MgFe': 'MgFe',
                'age_err': 'age_err',
                'meanRadkpc_kids': 'rad_kpc',
                'logM*': 'logM',
                'velDisp_ppxf_res': 'vdisp',
                #'velDisp_ppxf_err_res': 'vdisp_err',
                #'[M/H]': 'met',
                'dor_4': 'DoR',
                'met_err': 'met_err'
                } 

    train_df = train_df.rename(columns={old: new for old, new in train_mapping.items()})
    # train_df=train_df[columns]
    
    # age and met averaging
    test_df['lin_age'] = (test_df['age_unr']+test_df['age_rmax'])/2
    #test_df['logAge'] = np.log10(test_df['lin_age']) + 9 # 9 is the Gyr to log conversion I think 
    #test_df['logAge_err'] = np.log10(test_df['age_stdev']) + 9 
    
    test_df['met'] = (test_df['metal_unr']+test_df['metal_rmax'])/2
    
    
    snr_df = pd.read_csv('data/INSPIRE_SNR.csv')
    test_df = test_df.merge(snr_df[['ID_INSPIRE', 'SNR_MEAN']], on='ID_INSPIRE', how='left')
    # test_df['vdisp_err'] = np.where(test_df['SNR_MEAN'] > 30,test_df['Vdisp_XSH'] * 0.05,test_df['Vdisp_XSH'] * 0.10) 
    
    test_df.rename(columns={'ID_INSPIRE':'ID'}, inplace=True)

    # KIDS angle-> SDSS size 
    #test_df['Reff_median_SDSS_arcsec'] = ((test_df['Reff_median_KIDS_arcsec'] + 0.35)/0.86)**(2)
    test_df['rad_kpc'] = test_df.apply(lambda row: row['Reff_median_KIDS_arcsec'] * NedCalculator(z=row['zspec_XSH']).get_kpc_DA(), axis=1)
    
    test_mapping = {
            # 'lin_age': 'lin_age',
            # 'age_stdev': 'age_err',
            'age_stdev': 'lin_age_err',
            'rad_kpc': 'rad_kpc',
            'logMstarzspecTOT_S20': 'logM',
            'Vdisp_XSH': 'vdisp',
            #'vdisp_err': 'vdisp_err',
            'AlphaFe': 'MgFe',
            'met': 'met',
            'metal_stdev': 'met_err',
            'SNR_MEAN': 'SNR',
            }
    
    test_df = test_df.rename(columns={old: new for old, new in test_mapping.items()})
    
    
    #test_df['tau'] = (np.log10(test_df['tuni'])+9-test_df['logAge'])/(np.log10(test_df['tuni'])+9)
    #train_df['tau'] = (np.log10(train_df['univ_age'])+9-train_df['logAge'])/(np.log10(train_df['univ_age'])+9)
    
    test_df['tau'] = (test_df['tuni']-test_df['lin_age'])/(test_df['tuni'])
    train_df['tau'] = (train_df['univ_age']-train_df['lin_age'])/(train_df['univ_age'])
    
    
    print(test_df.columns.tolist())
    print(train_df.columns.tolist())
    
    print(train_df[columns].head(5))
    print("==============="*4)
    print(test_df[columns].head(5))
    
    print("Length of train df",len(train_df))
    
    if pc:
        test_df['lin_age_err'] = test_df['lin_age_err']/test_df['lin_age'] # make it percentage / ratios ratehr than absolute error
        train_df['lin_age_err'] = train_df['lin_age_err']/train_df['lin_age']
        
        test_df['met_err'] = test_df['met_err']/test_df['met']
        train_df['met_err'] = train_df['met_err']/train_df['met']
    
    train_df['Source'] = 1
    test_df['Source'] = 0

    if mix_datasets:
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        train_df, test_df = train_test_split(
            combined_df,
            test_size=test_size,
            random_state=random_state,
        )

    return train_df, test_df

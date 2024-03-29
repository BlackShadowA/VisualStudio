
import pandas as pd
import numpy as np
from typing import Union
import os
os.chdir('/Users/chentahung/Desktop/git/mob-py/src')
from PAV import PAV

class PAVA :
    def __init__(self, data, var :str, response : str, metric : str, add_var_aggFunc : dict = None, exclude_value = None) :
        '''
        Pool Adjacent Violators Algorithm (PAVA)
        
        data : <pandas.DataFrame> 
        var : <string> column for generate the x-axis of the greatest convex minorant (GCM)
        response : <string> column name of the value array created by response_func
        metric : <callable> The Statistics metric of the response variable
        add_var_aggFunc : <dict> Additional columns with corresponding statistics function (similar to pandas.DataFrame.agg())
        exclude_value : <None, list, float, int>
        sign : <str> {+|-|auto} decide the direction (correlation) of var and response variable.
        '''
        self._data = data
        self._var = var
        self._response = response
        self._metric = metric
        self._add_var_aggFunc = add_var_aggFunc
        self._exclude_value = exclude_value
        self._orgData_assignment = None
        self._CSD_Summary = None
        self._GCM_Summary = None
        self._PAV_Summary = None
        
        if self._data[self._var].isna().sum() > 0 :
            _isNaExist = True
        else :
            _isNaExist = False
              
        # check exclude values exist
        if isinstance(exclude_value, list) :
            if self._data[self._var].isin(exclude_value).sum() > 0 : #contains exclude value
                _isExcValueExist = True
            else :
                _isExcValueExist = False
        elif isinstance(exclude_value, (float, int)) :
            if self._data[self._var].isin([exclude_value]).sum() > 0 :
                _isExcValueExist = True
            else :
                _isExcValueExist = False
        elif exclude_value == None :
            _isExcValueExist = False
        else :
            _isExcValueExist = False
        
        '''
        self._df_sel : data that only contains clean values (no missing values and exclusive values)
        self._df_missing : data that the `var` variable is missing
        self._df_excvalue : data that the `var` variable is exclusive
        '''    
        
        if _isNaExist & _isExcValueExist : #both contains missing and exclude values
            self._df_missing = self._data.loc[self._data[self._var].isna(), :]
            
            if isinstance(exclude_value, list) :
                self._df_excvalue = self._data.loc[self._data[self._var].isin(exclude_value), :]
                self._df_sel = self._data.loc[(self._data[self._var].notnull()) & ~(self._data[self._var].isin(exclude_value))]
            elif isinstance(exclude_value, (float, int)) :
                self._df_excvalue = self._data.loc[self._data[self._var] == exclude_value, :]
                self._df_sel = self._data.loc[(self._data[self._var].notnull()) & (self._data[self._var] != exclude_value)]
            
        elif _isNaExist & ~_isExcValueExist: #only contain missing
            self._df_missing = self._data.loc[self._data[self._var].isna(), :]
            self._df_sel = self._data.loc[self._data[self._var].notnull()]
            
        elif ~_isNaExist & _isExcValueExist : #only contain exclude condition
            if isinstance(exclude_value, list) :
                self._df_excvalue = self._data.loc[self._data[self._var].isin(exclude_value), :]
                self._df_sel = self._data.loc[~self._data[self._var].isin(exclude_value)]
            elif isinstance(exclude_value, (float, int)) :
                self._df_excvalue = self._data.loc[self._data[self._var] == exclude_value, :]
                self._df_sel = self._data.loc[self._data[self._var] != exclude_value]
        
        else:
            self._df_sel = self._data
            
        self._isNaExist = _isNaExist
        self._isExcValueExist = _isExcValueExist
        
    @property
    def data(self) -> pd.DataFrame:
        return self._data
    
    @property
    def var(self) -> str :
        return self._var
    
    @property
    def response(self) -> str :
        return self._response
    
    @property
    def metric(self) -> str  :
        return self._metric

    @property
    def add_var_aggFunc(self) -> dict:
        return self._add_var_aggFunc
    
    @property
    def orgDataAssignment(self) -> Union[pd.DataFrame, None] :
        return self._orgData_assignment
    
    @property
    def GCM_Summary(self) -> Union[pd.DataFrame, None] :
        return self._GCM_Summary
    
    @property
    def CSD_Summary(self) -> Union[pd.DataFrame, None] :
        return self._CSD_Summary
    
    @property
    def PAV_Summary(self) -> Union[pd.DataFrame, None] :
        return self._PAV_Summary
    
    @property
    def df_missing(self) -> pd.DataFrame:
        return self._df_missing
    
    @property
    def df_excvalue(self) -> pd.DataFrame:
        return self._df_excvalue
    
    @property
    def df_sel(self) -> pd.DataFrame:
        return self._df_sel
    
    @property
    def isNaExist(self) -> bool :
        return self._isNaExist

    @property
    def isExcValueExist(self) -> bool :
        return self._isExcValueExist
        
    
    def __summarize_GCM_CSD(self, GCM, CSD) :
        '''
        CSD: acceptable inputs : { mean | sum | std | var | min | max | range }
            self.var | count | sum | std | max | min
            ----------------------------------------
        ===============================================================
        GCM : 
            intervalStart | intervalEnd | count | sum | std | max | min
            -----------------------------------------------------------
        '''

        _GCM = GCM.copy()
        if self.metric == 'mean' :
            _GCM[self.metric] = _GCM['sum']/_GCM['count']
            
        elif self.metric == 'var' :
            _GCM[self.metric] =  _GCM['std'] ** 2
                   
        elif self.metric == 'range' :
            _GCM[self.metric] =  _GCM['max'] - _GCM['min']
        
        else :
            pass
        
        _CSD = CSD.copy()
        _CSD['assignValue'] = _CSD.apply(lambda row: _GCM.loc[(GCM['intervalStart'] <= row[self.var]) & (_GCM['intervalEnd'] >= row[self.var]), 'intervalEnd'].values[0], axis=1)
        _CSD['assignMetric'] = _CSD.apply(lambda row: _GCM.loc[(GCM['intervalStart'] <= row[self.var]) & (_GCM['intervalEnd'] >= row[self.var]), self.metric].values[0], axis=1)

        
        return _GCM[['intervalStart', 'intervalEnd', self.metric]], _CSD[[self.var,  f'{self.response}_cum{self.metric.capitalize()}', 'assignValue', 'assignMetric']]
    

    def runPAVA(self, sign = 'auto') -> None:
        
        '''
        Execute PAVA Algorithm
        '''
        #Construct PAV object using the `df_sel` data which does not include missing value and exclusive values
        _PAV = PAV(data = self.df_sel, var = self.var, response = self.response, metric = self.metric)
        # PAVA is the Greatest convex minorant of the Cumulative Sum Diagram.
        # initialize the CSD (dataframe) and GCM (dataframe)
        # CSD is the initial groupby dataframe with stats information of response
        # GCM is the result the is tuned as monotonic on metric
        _GCM, _CSD = _PAV.init_CSD_GCM(sign = sign)       

        GCM, CSD = self.__summarize_GCM_CSD(GCM = _GCM, CSD = _CSD)
        
        # select column to generate additional variable aggrgation (add_var_aggFunc)
        chosen_col_list = [self.var, self.response] + list(self.add_var_aggFunc.keys())
        
        orgDataAssignment = self.df_sel.copy()[chosen_col_list]
        print(GCM)
        print('===================')
        print(CSD)
        print('===================')
        orgDataAssignment['assignValue'] = orgDataAssignment.apply(lambda row: GCM.loc[(GCM['intervalStart'] <= row[self.var]) & (GCM['intervalEnd'] >= row[self.var]), 'intervalStart'].values[0], axis=1)
        orgDataAssignment['assignMetric'] = orgDataAssignment.apply(lambda row: GCM.loc[(GCM['intervalStart'] <= row[self.var]) & (GCM['intervalEnd'] >= row[self.var]), self.metric].values[0], axis=1)
        print(orgDataAssignment)
        print(orgDataAssignment.dtypes)
        PAVA_Result_Df = orgDataAssignment.groupby('assignValue').agg(self.add_var_aggFunc).reset_index().fillna(0).sort_values(by='assignValue')
        newColName = ['intervalStart']
        for var, stat in zip(self.add_var_aggFunc.keys(), self.add_var_aggFunc.values()) :
            if isinstance(stat, list) :
                for singlestat in stat :
                    newColName.append('_'.join(tuple([var, singlestat])))
            else :
                newColName.append('_'.join(tuple([var, stat])))                
        PAVA_Result_Df.columns = newColName
        
        #Generate final PAVA result (include additional var aggregation functions)
        PAVA_Final_Df = GCM.sort_values(by = 'intervalStart').rename(columns = {self.metric : f'{self.response}_{self.metric}'})
        PAVA_Result = PAVA_Final_Df.merge(PAVA_Result_Df, how = 'left', on = 'intervalStart')
        PAVA_Result.iloc[0, 0] = -np.inf # continuous value range will have a -inf interval start at the smallest
        PAVA_Result.iloc[-1, 1] = np.inf # continuous value range will have a inf interval end at the greatest
        '''
        orgDataAssignment : original dataset but only select columns below 
        self.var | self.response | assignValue | assignMetric
        -----------------------------------------------------
        '''
        self._orgData_assignment = orgDataAssignment
        '''
        CSDSummary : a groupby(var) dataset with selected metric column which represent the cumulative sum diagram of the chosen variable and response
        self.var | self.metric
        ----------------------
        '''
        self._CSD_Summary = CSD
        '''
        GCM : the result of the greatest convex minorant diagram that no data violates the convexity.
        intervalStart | intervalEnd | mean
        ----------------------------------
        '''
        self._GCM_Summary = GCM
        '''
        PAVA_Result_Df: The result of PAVA contains additional variable statistics corresonding to the new given value of ɸ(x) to prevend convexity violation
        intervalStart | intervalEnd | mean | <add_var1>_<var_stat1> | <add_var2>_<var_stat1> | <add_var3>_<var_stat1> | <add_var3>_<var_stat2> | ... 
        --------------------------------------------------------------------------------------------------------------------------------------------
        '''
        self._PAV_Summary = PAVA_Result
        # TODO : PAVA self.var -> interval  `[intervalStart, intervarStart(shift(-1)))`

    # TODO
    def applyPAVA(self) -> pd.DataFrame :
        
        return None
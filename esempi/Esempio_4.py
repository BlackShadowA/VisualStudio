def runMOB(self, mergeMethod, sign = 'auto') -> pd.DataFrame :
    if mergeMethod in ['Stats', 'Size'] :
            
        outputTable = self.__summarizeBins(FinalOptTable = completeBinningTable)
    
    return outputTable
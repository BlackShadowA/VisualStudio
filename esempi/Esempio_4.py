def runMOB(self, mergeMethod, sign = 'auto') -> pd.DataFrame :
    if mergeMethod in ['Stats', 'Size'] :
            
        outputTable = self.__summarizeBins(FinalOptTable = completeBinningTable)
    
    return outputTable



from mrmr import mrmr_classif
selected_features = mrmr_classif(X=X, y=y, K=2)